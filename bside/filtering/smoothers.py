"""
Backward-pass smoothers that consume a forward-filter history and produce a
posterior estimate of the latent state given the *entire* observation
sequence.

Three families are provided:

* `RTSSmoother` -- Rauch-Tung-Striebel for linear-Gaussian systems.
* `UnscentedRTSSmoother` -- Sarkka (2008) unscented RTS for additive-Gaussian
  nonlinear systems.  Also accepts a Cubature or Gauss-Hermite predict rule.
* `ParticleSmoother` -- forward-filter / backward-simulation (Doucet et al. 2000).

All smoothers take the forward-filter history produced by
``filter(..., return_history=True)`` and the underlying `SSM`, and return a new
list of `FilteringDistribution` objects of the same length, containing the
smoothed marginals.
"""

from typing import List, Tuple
from math import log, pi

import torch
from torch import Tensor
from torch.linalg import solve_triangular

from bside.dynamics import AdditiveModel, LinearGaussianModel
from bside.ssm import SSM
from bside.filtering import functional as F
from bside.filtering.distributions import FilteringDistribution


# ---------------------------------------------------------------------------
# Backward step shared by RTS / URTS
# ---------------------------------------------------------------------------

def _smoothing_gain(
    U: Tensor,
    P_pred: Tensor,
    sqrt_P_pred: Tensor | None = None,
) -> Tensor:
    """
    Smoothing gain ``G = U P_pred^{-1}`` with ``U = Cov[x_k, x_{k+1} | y_{1:k}]``.

    Uses ``cholesky_solve`` when the Cholesky factor of ``P_pred`` is given,
    otherwise ``linalg.solve`` on the dense matrix (never an explicit inverse).
    """

    if sqrt_P_pred is not None:
        return torch.cholesky_solve(U.mT, sqrt_P_pred, upper=False).mT
    return torch.linalg.solve(P_pred, U, left=False)


def _rts_backward_step(
    dist_f: FilteringDistribution,
    dist_p: FilteringDistribution,
    U: Tensor,
    dist_s_next: FilteringDistribution,
) -> Tuple[FilteringDistribution, Tensor]:
    """
    One backward (RTS) step, common to every Gaussian smoother:

        G   = U P_pred^{-1}
        m^s = m + G (m^s_{k+1} - m_pred)
        P^s = P + G (P^s_{k+1} - P_pred) G^T

    The predicted moments ``(m_pred, P_pred)`` and cross-covariance ``U`` come
    from pushing the filtered distribution through the dynamics model; only
    that predict unit differs between RTS and its sigma-point variants.
    """

    G = _smoothing_gain(U, dist_p.cov)
    mean = dist_f.mean + torch.einsum('...ij,...j->...i', G, dist_s_next.mean - dist_p.mean)
    cov = dist_f.cov + G @ (dist_s_next.cov - dist_p.cov) @ G.mT
    return FilteringDistribution(mean=mean, cov=cov), G


def _input_at(u: Tensor | None, k: int) -> Tensor | None:
    return None if u is None else u[..., k, :]


class _GaussianSmoother:
    """
    Backward pass shared by the Gaussian smoothers.  Subclasses supply
    `_predict(dist_f, u)`, which pushes the filtered distribution at ``k``
    through the dynamics model and returns ``(p(x_{k+1} | y_{1:k}), U)``.
    """

    def _predict(
        self,
        dist_f: FilteringDistribution,
        u: Tensor | None,
    ) -> Tuple[FilteringDistribution, Tensor]:
        raise NotImplementedError

    def smooth(
        self,
        filtered_history: List[FilteringDistribution],
        u: Tensor | None = None,
        return_gains: bool = False,
    ) -> List[FilteringDistribution] | Tuple[List[FilteringDistribution], List[Tensor]]:
        """
        Parameters
        ----------
        filtered_history : list of FilteringDistribution
            Output of ``filter(..., return_history=True)``; entry ``k`` is
            ``p(x_k | y_{...k})``.  Batched histories are supported.
        u : Tensor, optional
            Inputs with shape ``(..., T, udim)`` -- pass ``data.u``.  Row ``k``
            drives the transition ``x_k -> x_{k+1}`` (the same convention the
            filter uses for either value of ``y0``).
        return_gains : bool
            Also return the smoothing gains ``G_k`` (``k = 0..K-1``); the lag-one
            smoothed cross-covariance is ``Cov[x_{k+1}, x_k | y] = P^s_{k+1} G_k^T``.
        """

        T = len(filtered_history)
        smoothed: List[FilteringDistribution] = [None] * T
        gains: List[Tensor] = [None] * max(T - 1, 0)
        smoothed[-1] = filtered_history[-1].snapshot()

        for k in range(T - 2, -1, -1):
            dist_f = filtered_history[k]
            dist_p, U = self._predict(dist_f, _input_at(u, k))
            smoothed[k], gains[k] = _rts_backward_step(dist_f, dist_p, U, smoothed[k + 1])

        return (smoothed, gains) if return_gains else smoothed


# ---------------------------------------------------------------------------
# RTS smoother (linear-Gaussian)
# ---------------------------------------------------------------------------

class RTSSmoother(_GaussianSmoother):
    """
    Rauch-Tung-Striebel smoother for a linear-Gaussian dynamics model.  The
    predict unit is the closed-form Kalman predict, which also yields the
    cross-covariance ``U = P A^T`` at no extra cost.
    """

    def __init__(
        self,
        model: SSM,
    ) -> None:

        if not isinstance(model.dynamics, LinearGaussianModel):
            raise ValueError(
                f"RTSSmoother requires a LinearGaussianModel for dynamics, got {type(model.dynamics)}."
            )
        self.model = model

    def _predict(self, dist_f, u):
        return F.kf_predict(self.model.dynamics, dist_f, u=u, crossCov=True)


# ---------------------------------------------------------------------------
# Unscented RTS smoother
# ---------------------------------------------------------------------------

class UnscentedRTSSmoother(_GaussianSmoother):
    """
    Unscented Rauch-Tung-Striebel smoother (Sarkka, 2008) for additive-noise
    dynamics.  The predict unit is the unscented transform: sigma points are
    formed around each filtered (mean, cov) and pushed through the dynamics.
    """

    def __init__(
        self,
        model: SSM,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:

        if not isinstance(model.dynamics, AdditiveModel):
            raise ValueError(
                "UnscentedRTSSmoother needs an AdditiveModel for dynamics so it can fold in process noise."
            )
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha**2 * (model.xdim + kappa) - model.xdim

    def _predict(self, dist_f, u):
        dist = dist_f.snapshot()
        dist.form_ut_weights(self.alpha, self.beta, self.kappa, self.lmbda)
        dist.form_ut_points(self.lmbda)
        return F.gaussian_quadrature(self.model.dynamics, dist, u=u, crossCov=True)


# ---------------------------------------------------------------------------
# Particle smoother (forward-filter / backward-simulation)
# ---------------------------------------------------------------------------

def _additive_gauss_log_transition(
    x_next: Tensor,
    x_curr: Tensor,
    dynamics: AdditiveModel,
    u: Tensor | None = None,
) -> Tensor:
    """
    Vectorized log p(x_next^j | x_curr^i) for an additive-Gaussian dynamics
    model. ``x_next`` has shape (J, d), ``x_curr`` has shape (I, d).
    Returns a (J, I) tensor of log-probabilities.
    """

    f_curr = dynamics(x_curr, u)  # (I, d)
    # residuals[j, i] = x_next[j] - f_curr[i]; shape (J, I, d)
    residuals = x_next.unsqueeze(1) - f_curr.unsqueeze(0)
    L_Q = dynamics.sqrt_noise_cov
    d = residuals.shape[-1]
    # solve_triangular over the last axis: treat residuals as ((J*I), d) batch
    flat = residuals.reshape(-1, d).T  # (d, J*I)
    sol = solve_triangular(L_Q, flat, upper=False)  # (d, J*I)
    mahal = torch.sum(sol * sol, dim=0).reshape(residuals.shape[:-1])  # (J, I)
    log_det = 2 * torch.sum(torch.log(torch.diagonal(L_Q)))
    return -0.5 * (mahal + log_det + d * log(2 * pi))


class ParticleSmoother:
    """
    Particle smoother via forward-filter / backward-simulation (FFBS).

    Requires a forward filter history with ``log_weights`` and ``particles``
    populated at every step (i.e. a `ParticleFilter` run with
    ``return_history=True``).  Dynamics must be an ``AdditiveModel`` so we can
    evaluate the transition density in closed form.
    """

    def __init__(
        self,
        model: SSM,
        n_samples: int | None = None,
    ) -> None:

        if not isinstance(model.dynamics, AdditiveModel):
            raise ValueError(
                "ParticleSmoother requires an AdditiveModel dynamics so log p(x_{t+1}|x_t) is tractable."
            )
        self.model = model
        self.n_samples = n_samples

    def smooth(
        self,
        filtered_history: List[FilteringDistribution],
    ) -> List[FilteringDistribution]:

        T = len(filtered_history)
        N = filtered_history[-1].size
        J = self.n_samples or N

        d = filtered_history[-1].particles.shape[-1]
        smoothed_particles = torch.zeros(T, J, d)

        # Step T-1: sample J indices from the final filtered weights.
        final = filtered_history[-1]
        idx = torch.multinomial(final.normalized_weights(), J, replacement=True)
        smoothed_particles[-1] = final.particles[idx]

        for t in range(T - 2, -1, -1):
            dist_t = filtered_history[t]
            log_w_t = dist_t.log_weights  # already normalized
            x_next = smoothed_particles[t + 1]  # (J, d)

            log_trans = _additive_gauss_log_transition(
                x_next, dist_t.particles, self.model.dynamics, u=None,
            )  # (J, N)

            # Per-trajectory backward weights: log_w_{i|j} = log_w_t[i] + log p(x_next^j | x_curr^i)
            log_back = log_trans + log_w_t.unsqueeze(0)  # (J, N)
            # Normalize across i for each j
            log_back = log_back - torch.logsumexp(log_back, dim=1, keepdim=True)
            back_w = torch.exp(log_back)  # (J, N)
            # Sample one ancestor per trajectory
            ancestors = torch.multinomial(back_w, 1, replacement=True).squeeze(-1)  # (J,)
            smoothed_particles[t] = dist_t.particles[ancestors]

        # Wrap into FilteringDistributions with equal smoothed weights
        smoothed: List[FilteringDistribution] = []
        log_w = -log(J) * torch.ones(J)
        for t in range(T):
            dist = FilteringDistribution(
                particles=smoothed_particles[t],
                log_weights=log_w.clone(),
            )
            dist.mean = torch.mean(smoothed_particles[t], dim=0)
            smoothed.append(dist)

        return smoothed
