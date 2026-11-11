"""
Expectation-maximization (EM) for linear-Gaussian state-space models.

Given an SSM with `LinearGaussianModel` dynamics and observations, EM iterates
between:

* E-step: Kalman filter + RTS smoother to obtain the smoothed marginals
  ``E[x_t | y_{1:T}]``, the smoothed covariances ``Cov[x_t | y_{1:T}]``, and
  the lag-one cross-covariances ``Cov[x_{t-1}, x_t | y_{1:T}]``.
* M-step: closed-form re-estimates of ``A``, ``C``, ``Q``, ``R``, ``mu_0``,
  ``P_0`` (Shumway & Stoffer, 1982).

The marginal-log-likelihood is reported every iteration; the iteration stops
once successive iterations differ by less than ``tol`` or after ``n_iter``
sweeps.

For nonlinear models the M-step has no closed form -- pair the EnKF/UKF/PF
forward pass with ``Posterior`` plus your favorite optimizer instead.
"""

from typing import Tuple

import torch
from torch import Tensor

from bside.dataset import Data
from bside.dynamics import LinearGaussianModel
from bside.filtering import FilteringDistribution, KalmanFilter
from bside.filtering.smoothers import RTSSmoother
from bside.ssm import SSM


def _second_moment(mean: Tensor, cov: Tensor) -> Tensor:
    """``E[x x^T] = P + m m^T`` (broadcast over leading dims)."""

    return cov + mean.unsqueeze(-1) * mean.unsqueeze(-2)


class EM:
    """
    Linear-Gaussian EM that mutates the matrices on an existing `SSM` in
    place.  The SSM's `LinearGaussianModel` dynamics / observations must wrap
    `LinearModel`s built from `Matrix` (or `PSDMatrix`) objects so the new
    A / C / Q / R can be written back via their ``val`` setters.

    The E-step is the forward Kalman filter followed by the RTS smoother; the
    M-step is the closed-form maximizer of the expected complete-data
    log-likelihood for the dynamics (A, Q), the observations (C, R) and the
    initial state (m_0, P_0).  Batched data ``(B, T, ydim)`` is supported:
    trajectories are independent and share all parameters.
    """

    def __init__(
        self,
        ssm: SSM,
        init_dist: FilteringDistribution,
        y0: bool = False,
    ) -> None:

        if not isinstance(ssm.dynamics, LinearGaussianModel):
            raise ValueError("EM requires a LinearGaussianModel for dynamics.")
        if not isinstance(ssm.observations, LinearGaussianModel):
            raise ValueError("EM requires a LinearGaussianModel for observations.")
        if ssm.dynamics.mat_u is not None or ssm.observations.mat_u is not None:
            raise NotImplementedError("EM does not yet estimate input matrices; use a model without `mat_u`.")

        self.ssm = ssm
        self.init_dist = init_dist.snapshot()
        self.y0 = y0

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward filter + backward RTS pass.

        Returns
        -------
        m_s : Tensor (..., K+1, xdim)
            Smoothed means of ``x_0 .. x_K``.
        P_s : Tensor (..., K+1, xdim, xdim)
            Smoothed covariances.
        P_lag : Tensor (..., K, xdim, xdim)
            Lag-one smoothed cross-covariances ``Cov[x_{k+1}, x_k | y]``.
        log_marg : Tensor
            Marginal log-likelihood ``log p(y | theta)`` (summed over a batch).
        """

        kf = KalmanFilter(model=self.ssm)
        filtered, log_marg = kf.filter(
            data=data,
            init_dist=self.init_dist,
            y0=self.y0,
            return_history=True,
            compute_log_prob=True,
        )
        smoothed, gains = RTSSmoother(self.ssm).smooth(filtered, u=data.u, return_gains=True)

        # Entries may have different batch shapes (an unbatched prior followed by
        # batched posteriors), so broadcast before stacking.
        batch = data.batch_shape
        d = self.ssm.xdim
        m_s = torch.stack([s.mean.expand(*batch, d) for s in smoothed], dim=-2)
        P_s = torch.stack([s.cov.expand(*batch, d, d) for s in smoothed], dim=-3)
        G = torch.stack([g.expand(*batch, d, d) for g in gains], dim=-3)
        P_lag = P_s[..., 1:, :, :] @ G.mT

        return m_s, P_s, P_lag, torch.as_tensor(log_marg).sum()

    # ------------------------------------------------------------------
    # M-step (closed form)
    # ------------------------------------------------------------------

    def _m_step(
        self,
        data: Data,
        m_s: Tensor,
        P_s: Tensor,
        P_lag: Tensor,
    ) -> None:

        y = data.y
        # Observed states: with y0 every state x_0..x_K has an observation;
        # otherwise x_0 is the unobserved prior and y row t observes x_{t+1}.
        m_obs = m_s if self.y0 else m_s[..., 1:, :]
        P_obs = P_s if self.y0 else P_s[..., 1:, :, :]

        def total(x: Tensor, event_dims: int = 2) -> Tensor:
            """Sum over every leading (batch and time) dimension."""
            return x.reshape(-1, *x.shape[-event_dims:]).sum(0)

        # Sufficient statistics
        Exx = _second_moment(m_s, P_s)                                   # E[x_k x_k^T]
        S00 = total(Exx[..., :-1, :, :])                                 # sum_{k<K}  E[x_k x_k^T]
        S11 = total(Exx[..., 1:, :, :])                                  # sum_{k>=1} E[x_k x_k^T]
        S10 = total(P_lag + m_s[..., 1:, :].unsqueeze(-1) * m_s[..., :-1, :].unsqueeze(-2))  # E[x_k x_{k-1}^T]
        Sxx = total(_second_moment(m_obs, P_obs))
        Syx = total(y.unsqueeze(-1) * m_obs.unsqueeze(-2))
        Syy = total(y.unsqueeze(-1) * y.unsqueeze(-2))

        n_trans = P_lag.reshape(-1, *P_lag.shape[-2:]).shape[0]
        n_obs = y.reshape(-1, y.shape[-1]).shape[0]

        # Dynamics: A = S10 S00^{-1},  Q = (S11 - A S10^T) / N
        A_new = torch.linalg.solve(S00, S10, left=False)
        Q_new = (S11 - A_new @ S10.mT - S10 @ A_new.mT + A_new @ S00 @ A_new.mT) / max(n_trans, 1)
        # Observations: C = Syx Sxx^{-1},  R = (Syy - C Syx^T) / N
        C_new = torch.linalg.solve(Sxx, Syx, left=False)
        R_new = (Syy - C_new @ Syx.mT - Syx @ C_new.mT + C_new @ Sxx @ C_new.mT) / n_obs

        # Symmetrize to guard against numerical asymmetry before Cholesky
        Q_new = 0.5 * (Q_new + Q_new.mT)
        R_new = 0.5 * (R_new + R_new.mT)

        # Write the new parameters back through the matrices' val setters.
        self.ssm.dynamics.model._mat_x.val = A_new
        self.ssm.dynamics._noise_cov.val = Q_new
        self.ssm.observations.model._mat_x.val = C_new
        self.ssm.observations._noise_cov.val = R_new

        # Initial state: average of the smoothed x_0 over the batch
        m0 = m_s[..., 0, :].reshape(-1, m_s.shape[-1])
        P0 = P_s[..., 0, :, :].reshape(-1, *P_s.shape[-2:])
        m0_new = m0.mean(0)
        r = m0 - m0_new
        P0_new = (P0 + r.unsqueeze(-1) * r.unsqueeze(-2)).mean(0)
        self.init_dist = FilteringDistribution(mean=m0_new, cov=0.5 * (P0_new + P0_new.mT))

    # ------------------------------------------------------------------
    # Driver loop
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Data,
        n_iter: int = 50,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> Tensor:
        """
        Run EM for at most ``n_iter`` sweeps or until the marginal log-likelihood
        improves by less than ``tol``.  Returns the per-iteration log-likelihood
        trajectory (1-D tensor).
        """

        log_marg_history = []
        prev_lm = -float('inf')

        for ii in range(n_iter):
            m_s, P_s, P_lag, log_marg = self._e_step(data)
            self._m_step(data, m_s, P_s, P_lag)

            log_marg_history.append(log_marg.detach().item())

            if verbose:
                print(f"EM iter {ii + 1:3d}: log p(y) = {log_marg_history[-1]:.4f}")

            if abs(log_marg_history[-1] - prev_lm) < tol:
                break
            prev_lm = log_marg_history[-1]

        return torch.tensor(log_marg_history)
