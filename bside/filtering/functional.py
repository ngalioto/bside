from typing import Tuple
import torch
from torch import Tensor
from torch.linalg import solve_triangular

from bside.filtering import FilteringDistribution
from bside.dynamics import Model, AdditiveModel, LinearGaussianModel


# ---------------------------------------------------------------------------
# Gain & update primitives shared by every Gaussian filter
# ---------------------------------------------------------------------------

def kalman_gain(
    dist: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor | None = None
) -> Tensor:
    
    """
    Compute the Kalman gain ``U @ S^-1`` using the cheapest factorization that
    is already up-to-date on ``dist``:

    * if ``Sinv`` is supplied, just multiply
    * else if the dense covariance is fresh, defer to ``torch.linalg.solve``
    * else use a two-stage triangular solve on the cached Cholesky factor
    """

    if Sinv is not None:
        K = U @ Sinv
    elif dist._cov is not None and not dist._cov._sqrt_up_to_date:
        K = torch.linalg.solve(dist.cov, U, left=False)
    else:
        K = solve_triangular(
            dist.sqrt_cov,
            solve_triangular(dist.sqrt_cov, U, upper=False, left=False),
            upper=False, left=False,
        )

    return K

def kalman_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    dist_y: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor | None = None
) -> FilteringDistribution:
    
    K = kalman_gain(dist_y, U, Sinv)

    dist_x.mean = dist_x.mean + torch.einsum('...ij, ...j -> ...i', K, (y - dist_y.mean))
    dist_x.cov = dist_x.cov - K @ U.T
    return dist_x

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
    Linear-Gaussian Kalman predict.  Computes ``U = Sigma A^T`` once and
    reuses it as both the cross-covariance and the building block for the
    propagated covariance ``A Sigma A^T + Q = A U + Q``.
    """

    U = dist.cov @ model.mat_x.T
    dist = FilteringDistribution(
        mean = model(dist.mean, u),
        cov = model.mat_x @ U + model.noise_cov
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
    
    dist_Y = FilteringDistribution(particles=model.sample(dist.particles, u))

    if crossCov:
        dist_Y.mean = torch.mean(dist_Y.particles, 0)
        res_Y = dist_Y.particles - dist_Y.mean
        dist_Y.cov = (res_Y.T @ res_Y) / (dist_Y.size - 1)

        if dist.mean is None:
            dist.mean = torch.mean(dist.particles, 0)
        U = ((dist.particles - dist.mean).T @ res_Y) / (dist.size - 1)

    return (dist_Y, U) if crossCov else dist_Y

def enkf_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    dist_y: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor = None
) -> FilteringDistribution:
        
    v = y - dist_y.particles
    K = kalman_gain(dist_y, U, Sinv)

    dist_x.particles = dist_x.particles + torch.einsum('ij, bj -> bi', K, v)

    dist_x.mean = None
    dist_x.cov = None

    return dist_x
    

# ---------------------------------------------------------------------------
# Deterministic-quadrature predict (UT / GH / Cubature)
# ---------------------------------------------------------------------------

def gaussian_quadrature(
    model: Model,
    dist_X: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False,
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
    
    """Propagate (particles, mean_weights, cov_weights) through ``model`` and
    aggregate the moments.  Works for any quadrature rule that supplies
    matching weights (unscented, Gauss-Hermite, cubature, etc.)."""

    additive = isinstance(model, AdditiveModel)

    Y = model(dist_X.particles, u) if additive else model.sample(dist_X.particles, u)

    Ymean = torch.sum(Y * dist_X.mean_weights.unsqueeze(-1), dim=0, keepdims=False)

    res_Y = Y - Ymean.unsqueeze(-2)
    P = (res_Y.T * dist_X.cov_weights) @ res_Y
    
    if additive:
        P = P + model.noise_cov

    dist_Y = FilteringDistribution(
        mean=Ymean, 
        cov=P, 
        particles=Y,
        quad_points=dist_X.quad_points,
        mean_weights=dist_X.mean_weights,
        cov_weights=dist_X.cov_weights
    )
        
    if crossCov:
        res_X = dist_X.particles - torch.sum(
            dist_X.particles.T * dist_X.mean_weights, 1, keepdims=True
        ).T
        U = (res_X.T * dist_X.cov_weights) @ res_Y
        return dist_Y, U
    
    else:
        return dist_Y


# ---------------------------------------------------------------------------
# Square-root Kalman primitives
#
# These never materialize the dense covariance; they manipulate Cholesky
# factors via QR factorizations of stacked sqrt-matrices.  The resulting
# Cholesky factors are returned with positive diagonals.
# ---------------------------------------------------------------------------

def _qr_chol(stacked: Tensor) -> Tensor:
    """
    Given a (n, m) matrix M with ``M @ M.T == Sigma``, return a lower-triangular
    Cholesky factor ``L`` of ``Sigma`` with positive diagonal.
    """
    # QR of M^T yields R upper triangular with R^T R = M M^T = Sigma.
    R = torch.linalg.qr(stacked.T, mode='reduced').R
    L = R.T
    # Flip column signs so the diagonal of L is positive.
    diag = torch.diagonal(L)
    sign = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    return L * sign.unsqueeze(0)


def srkf_predict(
    model: LinearGaussianModel,
    dist: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False,
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
    """
    Square-root Kalman predict.  Operates entirely on Cholesky factors:
    forms ``[A L_x | L_Q]`` and reads the new Cholesky factor off its QR.

    Cross-covariance ``U = Sigma A^T = L_x_pred (A L_x_pred^{-1}) ...`` — we
    return the dense ``U`` to keep the update step uniform with `kf_predict`,
    but it is computed from the freshly-propagated sqrt factor without ever
    forming a full ``L_x L_x^T``.
    """

    L_x = dist.sqrt_cov
    A = model.mat_x
    L_Q = model.sqrt_noise_cov

    AL = A @ L_x
    stacked = torch.cat([AL, L_Q], dim=1)
    L_x_pred = _qr_chol(stacked)

    new_dist = FilteringDistribution(
        mean=model(dist.mean, u),
        sqrt_cov=L_x_pred,
    )

    if crossCov:
        # U = Sigma_x_pred (no cross-product yet); we want Cov(x_pred, x_pred) A^T
        # is not the right cross-cov in the standard Kalman update -- here U is
        # used as a placeholder Cov(x_prior, x_pred). For the joint observation
        # update we need Sigma_pred H^T separately. Return None and let the
        # SR update compute it directly.
        return new_dist, None

    return new_dist


def srkf_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    model: LinearGaussianModel,
    u: Tensor | None = None,
    Sinv: Tensor | None = None,
) -> FilteringDistribution:
    """
    Square-root Kalman update.  Stacks the pre-array

        [ L_R^T          0       ]
        [ L_pred^T H^T   L_pred^T ]

    Applying QR triangularizes it to

        [ L_S^T   F^T          ]
        [   0     L_xpost^T    ]

    from which we read off the innovation-covariance Cholesky factor ``L_S``,
    the posterior-covariance Cholesky factor ``L_xpost``, and the gain
    ``K = F^T L_S^{-1}`` via a triangular solve.
    """

    H = model.mat_x
    L_R = model.sqrt_noise_cov
    L_pred = dist_x.sqrt_cov
    y_pred = model(dist_x.mean, u)

    xdim = L_pred.shape[0]
    ydim = L_R.shape[0]

    top = torch.cat([L_R.T, torch.zeros(ydim, xdim)], dim=1)
    bot = torch.cat([(H @ L_pred).T, L_pred.T], dim=1)
    pre_array = torch.cat([top, bot], dim=0)

    R = torch.linalg.qr(pre_array, mode='reduced').R
    # R is upper-triangular with the blocks we need on the diagonal:
    #   R[:ydim, :ydim] = L_S^T  -> innovation cov sqrt
    #   R[:ydim, ydim:] = F^T    -> K L_S^T
    #   R[ydim:, ydim:] = L_xpost^T
    L_S_T = R[:ydim, :ydim]
    F_T = R[:ydim, ydim:]
    L_xpost_T = R[ydim:, ydim:]

    # Fix signs so Cholesky factors have positive diagonals
    sign_S = torch.where(torch.diagonal(L_S_T) < 0, -1.0, 1.0)
    L_S_T = L_S_T * sign_S.unsqueeze(1)
    F_T = F_T * sign_S.unsqueeze(1)

    sign_x = torch.where(torch.diagonal(L_xpost_T) < 0, -1.0, 1.0)
    L_xpost_T = L_xpost_T * sign_x.unsqueeze(1)

    # K = F L_S^{-T} L_S^{-1}; with two triangular solves on L_S^T:
    # F = K L_S^T  ->  K = F (L_S^T)^{-1}
    K = solve_triangular(L_S_T, F_T, upper=True, left=True).T
    # K shape: (xdim, ydim)

    innovation = y - y_pred
    new_mean = dist_x.mean + torch.einsum('ij, j -> i', K, innovation)

    new_dist = FilteringDistribution(mean=new_mean, sqrt_cov=L_xpost_T.T)
    return new_dist


# ---------------------------------------------------------------------------
# Particle filter primitives
# ---------------------------------------------------------------------------

def bootstrap_pf_predict(
    model: Model,
    dist: FilteringDistribution,
    u: Tensor = None,
) -> FilteringDistribution:
    """Bootstrap proposal: sample x_t^i ~ p(x_t | x_{t-1}^i)."""

    new_particles = model.sample(dist.particles, u)
    new_dist = FilteringDistribution(
        particles=new_particles,
        log_weights=dist.log_weights.clone() if dist.log_weights is not None else None,
    )
    return new_dist


def particle_log_likelihood(
    y: Tensor,
    particles: Tensor,
    observation_model: AdditiveModel,
) -> Tensor:
    """
    Per-particle log-likelihood log p(y | x_i) for an additive-Gaussian
    observation model: ``y - h(x_i) ~ N(0, R)``.  Uses solve_triangular on
    the Cholesky factor (never forms R^-1).
    """

    h = observation_model(particles, None) if hasattr(observation_model, 'model') else observation_model(particles)
    residuals = y.unsqueeze(0) - h  # (N, ydim)
    L_R = observation_model.sqrt_noise_cov
    # Mahalanobis term per particle
    sol = solve_triangular(L_R, residuals.T, upper=False)  # (ydim, N)
    mahal = torch.sum(sol * sol, dim=0)  # (N,)
    log_det = 2 * torch.sum(torch.log(torch.diagonal(L_R)))
    from math import log as _log, pi as _pi
    ydim = y.shape[-1]
    return -0.5 * (mahal + log_det + ydim * _log(2 * _pi))


def particle_filter_update(
    y: Tensor,
    dist: FilteringDistribution,
    observation_model: AdditiveModel,
    resample_threshold: float | None = 0.5,
    resample_method: str = 'systematic',
) -> Tuple[FilteringDistribution, Tensor]:
    """
    Bootstrap-PF measurement update.  Returns the updated distribution and the
    per-step log-marginal-likelihood contribution
    ``log p(y_t | y_{1:t-1}) = logsumexp(log_w_prev + log p(y|x_i))``.
    """

    log_lik = particle_log_likelihood(y, dist.particles, observation_model)
    prev_log_w = dist.log_weights
    if prev_log_w is None:
        from math import log as _log
        prev_log_w = -_log(dist.size) * torch.ones(dist.size)
    new_log_w_unnorm = prev_log_w + log_lik
    # log-marginal-likelihood contribution = logsumexp of unnormalized log weights.
    log_marg = torch.logsumexp(new_log_w_unnorm, dim=0)
    # Normalize for storage
    dist.log_weights = new_log_w_unnorm - log_marg

    if resample_threshold is not None:
        ess = dist.effective_sample_size()
        if ess < resample_threshold * dist.size:
            dist.resample(method=resample_method)

    return dist, log_marg
