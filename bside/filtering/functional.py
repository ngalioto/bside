"""
Stateless filtering primitives.

Every Bayesian filter here is the same recursion over two units:

* **predict** -- push the current state distribution through the *dynamics*
  model ``x_k = Psi(x_{k-1}, u) + xi_k``: the deterministic map moves the
  mean / points, the stochastic part adds the process-noise covariance.
* **update** -- push the predicted distribution through the *observation*
  model ``y_k = h(x_k, u) + eta_k`` to get the predictive density
  ``p(y_k | y_{1:k-1})`` (whose value at the data is the marginal-likelihood
  term), then condition on ``y_k``.

The functions below implement those two units for each approximation family
(closed form, square root, deterministic quadrature, ensemble, particle).
All Gaussian primitives broadcast over leading batch dimensions, so a batch of
independent trajectories is filtered in one pass.
"""

from math import log, pi
from typing import Tuple

import torch
from torch import Tensor
from torch.linalg import solve_triangular

from bside.filtering.distributions import FilteringDistribution
from bside.dynamics import Model, AdditiveModel, LinearGaussianModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_log_prob_from_sqrt(
    v: Tensor,
    L: Tensor,
) -> Tensor:
    """``log N(v; 0, L L^T)`` via a triangular solve (no explicit inverse)."""

    z = solve_triangular(L, v.unsqueeze(-1), upper=False).squeeze(-1)
    log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
    return -0.5 * (torch.sum(z * z, dim=-1) + log_det + v.shape[-1] * log(2 * pi))


def _evaluate(
    model: Model,
    X: Tensor,
    u: Tensor | None = None,
    stochastic: bool = False,
) -> Tensor:
    """
    Evaluate ``model`` at a set of points ``X`` with shape ``(..., N, d)``.

    The point axis is moved to the front while the model is called, so model
    parameters with batch dimensions (a batch of parameter vectors, see
    `bside.sysid.Posterior`) broadcast against the remaining leading dimensions.
    Model maps must therefore act on the last axis and broadcast over any
    leading ones.  The input ``u`` (shape ``(..., udim)``) is shared by every
    point.  ``stochastic=True`` draws from the model (deterministic part +
    noise) instead of evaluating the deterministic part.
    """

    Xm = X.movedim(-2, 0)
    um = None if u is None else u.unsqueeze(0).expand(*Xm.shape[:-1], u.shape[-1])
    out = model.sample(Xm, um) if stochastic else model(Xm, um)
    return out.movedim(0, -2)


def _broadcast_batch(*mats: Tensor) -> list[Tensor]:
    """Expand matrices ``(..., n, m)`` to a common batch shape."""

    batch = torch.broadcast_shapes(*(M.shape[:-2] for M in mats))
    return [M.expand(*batch, *M.shape[-2:]) for M in mats]


def _positive_diagonal(
    R: Tensor,
) -> Tensor:
    """Row signs that make the diagonal of an upper-triangular ``R`` positive."""

    return torch.where(torch.diagonal(R, dim1=-2, dim2=-1) < 0, -1.0, 1.0).to(R.dtype)


# ---------------------------------------------------------------------------
# Gain & update primitives shared by every Gaussian filter
# ---------------------------------------------------------------------------

def kalman_gain(
    dist: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor | None = None
) -> Tensor:

    """
    Kalman gain ``K = U S^{-1}``, where ``S`` is the covariance of ``dist``
    (the predicted observation), using the cheapest factorization available:

    * ``Sinv`` supplied -> one matmul
    * Cholesky factor of ``S`` already cached (e.g. after evaluating the
      likelihood) -> ``cholesky_solve``
    * otherwise -> ``torch.linalg.solve`` on the dense ``S``
    """

    if Sinv is not None:
        return U @ Sinv
    if dist._cov._sqrt_up_to_date:
        # K^T = S^{-1} U^T with S = L L^T
        return torch.cholesky_solve(U.mT, dist.sqrt_cov, upper=False).mT
    return torch.linalg.solve(dist.cov, U, left=False)


def kalman_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    dist_y: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor | None = None
) -> FilteringDistribution:

    """
    Gaussian conditioning on ``y`` given the joint moments of (x, y):

        K = U S^{-1},   m = m^- + K (y - mu_y),   P = P^- - K U^T

    Returns a new distribution; the inputs are not modified.  Quadrature
    weights are carried over so the next predict can re-use them.
    """

    K = kalman_gain(dist_y, U, Sinv)

    return FilteringDistribution(
        mean=dist_x.mean + torch.einsum('...ij,...j->...i', K, y - dist_y.mean),
        cov=dist_x.cov - K @ U.mT,
        quad_points=dist_x.quad_points,
        mean_weights=dist_x.mean_weights,
        cov_weights=dist_x.cov_weights,
    )


# ---------------------------------------------------------------------------
# Standard Kalman / EnKF predict & update
# ---------------------------------------------------------------------------

def kf_predict(
    model: LinearGaussianModel,
    dist: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False,
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:

    """
    Push a Gaussian through a linear-Gaussian model ``A x + B u + noise``.

    Computes ``U = P A^T`` once and reuses it both as the cross-covariance and
    for the propagated covariance ``A P A^T + Q = A U + Q``.
    """

    U = dist.cov @ model.mat_x.mT
    dist = FilteringDistribution(
        mean=model(dist.mean, u),
        cov=model.mat_x @ U + model.noise_cov
    )

    return (dist, U) if crossCov else dist


def enkf_predict(
    model: Model,
    dist: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:

    """Ensemble Kalman predict: draws fresh particles via ``model.sample`` and,
    if requested, returns the propagated mean / cov together with the
    cross-covariance between the input and propagated particles.
    """

    dist_Y = FilteringDistribution(particles=_evaluate(model, dist.particles, u, stochastic=True))

    if crossCov:
        dist_Y.mean = torch.mean(dist_Y.particles, 0)
        res_Y = dist_Y.particles - dist_Y.mean
        dist_Y.cov = (res_Y.T @ res_Y) / (dist_Y.size - 1)

        mean_X = dist.mean if dist.mean is not None else torch.mean(dist.particles, 0)
        U = ((dist.particles - mean_X).T @ res_Y) / (dist.size - 1)
        return dist_Y, U

    return dist_Y


def enkf_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    dist_y: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor = None
) -> FilteringDistribution:

    """Stochastic EnKF update: shift each member by ``K (y - y_i)``."""

    K = kalman_gain(dist_y, U, Sinv)
    v = y - dist_y.particles

    return FilteringDistribution(particles=dist_x.particles + v @ K.T)


# ---------------------------------------------------------------------------
# Deterministic-quadrature predict (UT / GH / Cubature)
# ---------------------------------------------------------------------------

def gaussian_quadrature(
    model: Model,
    dist_X: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False,
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:

    """
    Push quadrature points through ``model`` and aggregate the moments.

    The deterministic part of the model moves the points; for an additive
    model the stochastic part contributes its noise covariance exactly.
    Works for any rule that supplies (particles, mean_weights, cov_weights):
    unscented, Gauss-Hermite, cubature.  Particles have shape ``(..., N, d)``.
    """

    additive = isinstance(model, AdditiveModel)
    wm, wc = dist_X.mean_weights, dist_X.cov_weights

    Y = _evaluate(model, dist_X.particles, u, stochastic=not additive)
    Ymean = wm @ Y
    res_Y = Y - Ymean.unsqueeze(-2)
    P = (res_Y.mT * wc) @ res_Y

    if additive:
        P = P + model.noise_cov

    dist_Y = FilteringDistribution(
        mean=Ymean,
        cov=P,
        particles=Y,
        quad_points=dist_X.quad_points,
        mean_weights=wm,
        cov_weights=wc,
    )

    if crossCov:
        Xmean = wm @ dist_X.particles
        res_X = dist_X.particles - Xmean.unsqueeze(-2)
        U = (res_X.mT * wc) @ res_Y
        return dist_Y, U

    return dist_Y


# ---------------------------------------------------------------------------
# Square-root Kalman primitives
#
# These never materialize a dense covariance; they manipulate Cholesky
# factors via QR factorizations of stacked square-root matrices.  Resulting
# Cholesky factors have positive diagonals.
# ---------------------------------------------------------------------------

def _qr_chol(stacked: Tensor) -> Tensor:
    """
    Given ``M`` with shape ``(..., n, m)`` and ``M M^T = Sigma``, return the
    lower-triangular Cholesky factor of ``Sigma`` with positive diagonal.
    """

    # QR of M^T yields upper-triangular R with R^T R = M M^T.
    R = torch.linalg.qr(stacked.mT, mode='reduced').R
    return (R * _positive_diagonal(R).unsqueeze(-1)).mT


def srkf_predict(
    model: LinearGaussianModel,
    dist: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False,
) -> Tuple[FilteringDistribution, None] | FilteringDistribution:
    """
    Square-root predict through a linear-Gaussian model: the new Cholesky
    factor is read off the QR of ``[A L_x | L_Q]``.

    The square-root update forms its own cross terms, so ``crossCov=True``
    returns ``None`` in place of a cross-covariance.
    """

    AL, L_Q = _broadcast_batch(model.mat_x @ dist.sqrt_cov, model.sqrt_noise_cov)
    new_dist = FilteringDistribution(
        mean=model(dist.mean, u),
        sqrt_cov=_qr_chol(torch.cat([AL, L_Q], dim=-1)),
    )

    return (new_dist, None) if crossCov else new_dist


def srkf_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    model: LinearGaussianModel,
    u: Tensor | None = None,
    compute_log_prob: bool = False,
) -> Tuple[FilteringDistribution, Tensor | None]:
    """
    Square-root update through a linear-Gaussian observation model.  QR of the
    pre-array

        [ L_R^T          0       ]
        [ L_pred^T H^T   L_pred^T ]

    gives the upper-triangular

        [ L_S^T   F^T          ]
        [   0     L_xpost^T    ]

    with innovation-covariance factor ``L_S``, posterior factor ``L_xpost``
    and gain ``K = F L_S^{-1}``.  The marginal-likelihood term
    ``log N(y; y_pred, S)`` is evaluated from ``L_S`` directly, so the dense
    innovation covariance is never formed.

    Returns ``(posterior, log_lik)``; ``log_lik`` is None unless requested.
    """

    H = model.mat_x
    HL, L_pred, L_R = _broadcast_batch(H @ dist_x.sqrt_cov, dist_x.sqrt_cov, model.sqrt_noise_cov)
    batch = L_pred.shape[:-2]
    xdim, ydim = L_pred.shape[-1], L_R.shape[-1]

    top = torch.cat([L_R.mT, L_R.new_zeros(*batch, ydim, xdim)], dim=-1)
    bot = torch.cat([HL.mT, L_pred.mT], dim=-1)
    R = torch.linalg.qr(torch.cat([top, bot], dim=-2), mode='reduced').R

    L_S_T, F_T, L_xpost_T = R[..., :ydim, :ydim], R[..., :ydim, ydim:], R[..., ydim:, ydim:]

    sign_S = _positive_diagonal(L_S_T).unsqueeze(-1)
    L_S_T, F_T = L_S_T * sign_S, F_T * sign_S
    L_xpost_T = L_xpost_T * _positive_diagonal(L_xpost_T).unsqueeze(-1)

    # F^T = L_S^T K^T  ->  K = (L_S^{-T} F^T)^T
    K = solve_triangular(L_S_T, F_T, upper=True).mT

    v = y - model(dist_x.mean, u)  # innovation
    posterior = FilteringDistribution(
        mean=dist_x.mean + torch.einsum('...ij,...j->...i', K, v),
        sqrt_cov=L_xpost_T.mT,
    )
    log_lik = _gaussian_log_prob_from_sqrt(v, L_S_T.mT) if compute_log_prob else None

    return posterior, log_lik


# ---------------------------------------------------------------------------
# Particle filter primitives
# ---------------------------------------------------------------------------

def bootstrap_pf_predict(
    model: Model,
    dist: FilteringDistribution,
    u: Tensor = None,
) -> FilteringDistribution:
    """Bootstrap proposal: sample x_t^i ~ p(x_t | x_{t-1}^i)."""

    return FilteringDistribution(
        particles=_evaluate(model, dist.particles, u, stochastic=True),
        log_weights=dist.log_weights,
    )


def particle_log_likelihood(
    y: Tensor,
    particles: Tensor,
    observation_model: AdditiveModel,
    u: Tensor | None = None,
) -> Tensor:
    """
    Per-particle log-likelihood ``log p(y | x_i)`` for an additive-Gaussian
    observation model ``y - h(x_i) ~ N(0, R)``.
    """

    residuals = y - _evaluate(observation_model, particles, u)  # (N, ydim)
    return _gaussian_log_prob_from_sqrt(residuals, observation_model.sqrt_noise_cov)


def particle_filter_update(
    y: Tensor,
    dist: FilteringDistribution,
    observation_model: AdditiveModel,
    resample_threshold: float | None = 0.5,
    resample_method: str = 'systematic',
    u: Tensor | None = None,
) -> Tuple[FilteringDistribution, Tensor]:
    """
    Bootstrap-PF measurement update.  Returns the updated distribution and the
    per-step log-marginal-likelihood contribution
    ``log p(y_t | y_{1:t-1}) = logsumexp(log_w_prev + log p(y|x_i))``.
    """

    log_lik = particle_log_likelihood(y, dist.particles, observation_model, u)
    prev_log_w = dist.log_weights
    if prev_log_w is None:
        prev_log_w = torch.full((dist.size,), -log(dist.size))
    new_log_w_unnorm = prev_log_w + log_lik
    log_marg = torch.logsumexp(new_log_w_unnorm, dim=0)

    new_dist = FilteringDistribution(
        particles=dist.particles,
        log_weights=new_log_w_unnorm - log_marg,
    )

    if resample_threshold is not None:
        if new_dist.effective_sample_size() < resample_threshold * new_dist.size:
            new_dist.resample(method=resample_method)

    return new_dist, log_marg
