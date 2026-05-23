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

import copy
from typing import List
from math import log, pi

import torch
from torch import Tensor
from torch.linalg import solve_triangular

from bside.dynamics import AdditiveModel, LinearGaussianModel, Model
from bside.ssm import SSM
from bside.filtering import functional as F
from bside.filtering.distributions import FilteringDistribution


# ---------------------------------------------------------------------------
# Gain helper shared by RTS / URTS
# ---------------------------------------------------------------------------

def _smoothing_gain(
    U: Tensor,
    P_pred: Tensor,
    sqrt_P_pred: Tensor | None = None,
) -> Tensor:
    """
    Smoothing gain ``C = U P_pred^{-1}``.

    Uses two triangular solves on the Cholesky factor when available
    (avoids forming P_pred^{-1}), otherwise falls back to ``linalg.solve``
    on the dense matrix.
    """

    if sqrt_P_pred is not None:
        return solve_triangular(
            sqrt_P_pred,
            solve_triangular(sqrt_P_pred, U, upper=False, left=False),
            upper=False, left=False,
        )
    return torch.linalg.solve(P_pred, U, left=False)


# ---------------------------------------------------------------------------
# RTS smoother (linear-Gaussian)
# ---------------------------------------------------------------------------

class RTSSmoother:
    """
    Rauch-Tung-Striebel smoother for a linear-Gaussian dynamics model.

    Re-uses the ``U = Sigma A^T`` matrix that the Kalman predict already
    computes, so the smoothing gain ``C = U P_pred^{-1}`` does not require any
    extra matrix multiplications.
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

    def smooth(
        self,
        filtered_history: List[FilteringDistribution],
    ) -> List[FilteringDistribution]:

        T = len(filtered_history)
        smoothed: List[FilteringDistribution] = [None] * T
        smoothed[-1] = copy.deepcopy(filtered_history[-1])

        for t in range(T - 2, -1, -1):
            dist_f = filtered_history[t]
            # Run the linear-Gaussian predict once to obtain (x_pred, P_pred, U).
            dist_p, U = F.kf_predict(self.model.dynamics, dist_f, u=None, crossCov=True)
            C = _smoothing_gain(U, dist_p.cov)
            mean_s = dist_f.mean + torch.einsum('ij,j->i', C, smoothed[t + 1].mean - dist_p.mean)
            cov_s = dist_f.cov + C @ (smoothed[t + 1].cov - dist_p.cov) @ C.T
            smoothed[t] = FilteringDistribution(mean=mean_s, cov=cov_s)

        return smoothed


# ---------------------------------------------------------------------------
# Unscented RTS smoother (and Cubature / Gauss-Hermite variants via the same predict)
# ---------------------------------------------------------------------------

class UnscentedRTSSmoother:
    """
    Unscented Rauch-Tung-Striebel smoother (Sarkka, 2008).

    Works for any nonlinear additive-Gaussian dynamics model.  The
    quadrature rule is configurable via ``predict_rule`` which must accept
    ``(model, dist, u, crossCov)`` and return ``(predicted_dist, cross_cov)``
    when ``crossCov=True``; the default uses standard UT sigma points.
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

    def smooth(
        self,
        filtered_history: List[FilteringDistribution],
    ) -> List[FilteringDistribution]:

        T = len(filtered_history)
        smoothed: List[FilteringDistribution] = [None] * T
        smoothed[-1] = copy.deepcopy(filtered_history[-1])

        for t in range(T - 2, -1, -1):
            dist_f = copy.deepcopy(filtered_history[t])
            # Recompute UT weights & points around the filtered (mean, cov) and
            # propagate through dynamics so we can read off the cross-covariance.
            dist_f.form_ut_weights(self.alpha, self.beta, self.kappa, self.lmbda)
            dist_f.form_ut_points(self.lmbda)
            dist_p, U = F.gaussian_quadrature(self.model.dynamics, dist_f, u=None, crossCov=True)
            C = _smoothing_gain(U, dist_p.cov)
            mean_s = dist_f.mean + torch.einsum('ij,j->i', C, smoothed[t + 1].mean - dist_p.mean)
            cov_s = dist_f.cov + C @ (smoothed[t + 1].cov - dist_p.cov) @ C.T
            smoothed[t] = FilteringDistribution(mean=mean_s, cov=cov_s)

        return smoothed


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
