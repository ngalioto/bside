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

import copy
from typing import Tuple

import torch
from torch import Tensor

from bside.dataset import Data
from bside.dynamics import LinearGaussianModel
from bside.filtering import FilteringDistribution, KalmanFilter
from bside.filtering.smoothers import _smoothing_gain
from bside.filtering import functional as F
from bside.ssm import SSM


class EM:
    """
    Linear-Gaussian EM that mutates the matrices on an existing `SSM` in
    place.  The SSM's `LinearGaussianModel` dynamics / observations must wrap
    `LinearModel`s built from `Matrix` (or `PSDMatrix`) objects so the new
    A / C / Q / R can be written back via their ``val`` setters.
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

        self.ssm = ssm
        self.init_dist = copy.deepcopy(init_dist)
        self.y0 = y0

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Run the forward filter and the backward RTS pass; return everything the
        M-step needs.

        Returns
        -------
        mu_s : Tensor (T, xdim)
            Smoothed state means.
        P_s : Tensor (T, xdim, xdim)
            Smoothed state covariances.
        P_lag : Tensor (T-1, xdim, xdim)
            Smoothed lag-one cross-covariances ``Cov[x_{t-1}, x_t | y]``.
        log_marg : Tensor
            Forward-filter marginal log-likelihood ``log p(y_{1:T})``.
        """

        kf = KalmanFilter(model=self.ssm)
        filtered_history, log_marg = kf.filter(
            data=data,
            init_dist=self.init_dist,
            y0=self.y0,
            return_history=True,
            compute_log_prob=True,
        )

        T = len(filtered_history)
        xdim = self.ssm.xdim
        mu_s = torch.zeros(T, xdim)
        P_s = torch.zeros(T, xdim, xdim)
        P_lag = torch.zeros(max(T - 1, 0), xdim, xdim)

        # Backward RTS: walk from T-1 to 0, recording the smoothing gains so we
        # can compute the lag-one cross-covariances in the same pass.
        mu_s[-1] = filtered_history[-1].mean
        P_s[-1] = filtered_history[-1].cov

        # We will need the *smoother gains* C_t to compute the lag-one cov:
        #     Cov[x_t, x_{t+1} | y_{1:T}] = C_t P_s[t+1]
        # Store the gains as we go.
        gains = [None] * (T - 1)

        for t in range(T - 2, -1, -1):
            dist_f = filtered_history[t]
            dist_p, U = F.kf_predict(self.ssm.dynamics, dist_f, u=None, crossCov=True)
            C = _smoothing_gain(U, dist_p.cov)
            gains[t] = C
            mu_s[t] = dist_f.mean + torch.einsum('ij,j->i', C, mu_s[t + 1] - dist_p.mean)
            P_s[t] = dist_f.cov + C @ (P_s[t + 1] - dist_p.cov) @ C.T

        for t in range(T - 1):
            P_lag[t] = gains[t] @ P_s[t + 1]

        return mu_s, P_s, P_lag, log_marg

    # ------------------------------------------------------------------
    # M-step (Shumway & Stoffer closed form)
    # ------------------------------------------------------------------

    def _m_step(
        self,
        data: Data,
        mu_s: Tensor,
        P_s: Tensor,
        P_lag: Tensor,
    ) -> None:

        y = data.y
        T = mu_s.shape[0]

        # Sufficient statistics
        # S_xx_lag_to = sum_{t=1..T-1} E[x_t x_t^T]
        # S_xx_lag_from = sum_{t=0..T-2} E[x_{t-1} x_{t-1}^T]
        # S_xx_pair = sum_{t=1..T-1} E[x_t x_{t-1}^T]
        S_xx_lag_to = torch.zeros(self.ssm.xdim, self.ssm.xdim)
        S_xx_lag_from = torch.zeros(self.ssm.xdim, self.ssm.xdim)
        S_xx_pair = torch.zeros(self.ssm.xdim, self.ssm.xdim)
        S_xx_all = torch.zeros(self.ssm.xdim, self.ssm.xdim)
        S_yx = torch.zeros(self.ssm.ydim, self.ssm.xdim)
        S_yy = torch.zeros(self.ssm.ydim, self.ssm.ydim)

        for t in range(T):
            S_xx_all = S_xx_all + P_s[t] + torch.outer(mu_s[t], mu_s[t])
            S_yx = S_yx + torch.outer(y[t], mu_s[t])
            S_yy = S_yy + torch.outer(y[t], y[t])

        for t in range(1, T):
            S_xx_lag_to = S_xx_lag_to + P_s[t] + torch.outer(mu_s[t], mu_s[t])
            S_xx_lag_from = S_xx_lag_from + P_s[t - 1] + torch.outer(mu_s[t - 1], mu_s[t - 1])
            S_xx_pair = S_xx_pair + P_lag[t - 1] + torch.outer(mu_s[t], mu_s[t - 1])

        # Closed-form M-step updates
        A_new = torch.linalg.solve(S_xx_lag_from.T, S_xx_pair.T).T
        C_new = torch.linalg.solve(S_xx_all.T, S_yx.T).T
        Q_new = (S_xx_lag_to - A_new @ S_xx_pair.T - S_xx_pair @ A_new.T + A_new @ S_xx_lag_from @ A_new.T) / max(T - 1, 1)
        R_new = (S_yy - C_new @ S_yx.T - S_yx @ C_new.T + C_new @ S_xx_all @ C_new.T) / T

        # Symmetrize to guard against numerical asymmetry before Cholesky
        Q_new = 0.5 * (Q_new + Q_new.T)
        R_new = 0.5 * (R_new + R_new.T)

        # Write the new parameters back through the matrices' val setters.
        self.ssm.dynamics.model._mat_x.val = A_new
        self.ssm.dynamics._noise_cov.val = Q_new
        self.ssm.observations.model._mat_x.val = C_new
        self.ssm.observations._noise_cov.val = R_new

        # Refresh the initial-state distribution
        self.init_dist = FilteringDistribution(mean=mu_s[0].clone(), cov=P_s[0].clone())

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
            mu_s, P_s, P_lag, log_marg = self._e_step(data)
            self._m_step(data, mu_s, P_s, P_lag)

            log_marg_history.append(log_marg.detach().item() if isinstance(log_marg, Tensor) else float(log_marg))

            if verbose:
                print(f"EM iter {ii + 1:3d}: log p(y) = {log_marg_history[-1]:.4f}")

            if abs(log_marg_history[-1] - prev_lm) < tol:
                break
            prev_lm = log_marg_history[-1]

        return torch.tensor(log_marg_history)
