"""
Parallel-in-time Kalman filtering and RTS smoothing.

Särkkä & García-Fernández (2021), "Temporal parallelization of Bayesian
smoothers", IEEE TAC.  Each time step of the linear-Gaussian filter (and of
the RTS smoother) is written as an element of an associative operator, so the
whole recursion is a prefix scan: ``log2(T)`` vectorized passes instead of
``T`` sequential Python steps.  Results match `KalmanFilter` / `RTSSmoother`
(up to floating point), including batched trajectories, batched parameters
and missing data.

The model must be time-invariant linear-Gaussian (inputs enter additively
through ``mat_u``), which is what `KalmanFilter` assumes as well.
"""

from typing import Callable, List, Tuple

import torch
from torch import Tensor

from bside.dataset import Data
from bside.filtering.distributions import FilteringDistribution
from bside.filtering.filters import KalmanFilter
from bside.filtering.functional import _gaussian_log_prob_from_sqrt
from bside.filtering.smoothers import RTSSmoother

Elements = Tuple[Tensor, ...]


# ---------------------------------------------------------------------------
# Generic scan (time axis first)
# ---------------------------------------------------------------------------

def _scan(
    elems: Elements,
    op: Callable[[Elements, Elements], Elements],
    reverse: bool = False,
) -> Elements:
    """
    Inclusive scan along dim 0 with an associative ``op(earlier, later)``
    (Hillis-Steele: ``ceil(log2 T)`` vectorized passes).  ``reverse=True``
    gives suffix aggregates, as the smoother needs.
    """

    if reverse:
        elems = tuple(e.flip(0) for e in elems)

    T, offset = elems[0].shape[0], 1
    while offset < T:
        before = tuple(e[:-offset] for e in elems)
        after = tuple(e[offset:] for e in elems)
        # In flipped order the earlier index is the later time.
        merged = op(after, before) if reverse else op(before, after)
        elems = tuple(torch.cat([e[:offset], m]) for e, m in zip(elems, merged))
        offset *= 2

    return tuple(e.flip(0) for e in elems) if reverse else elems


def _mv(M: Tensor, v: Tensor) -> Tensor:
    return (M @ v.unsqueeze(-1)).squeeze(-1)


def _offset(model, u: Tensor | None, like: Tensor) -> Tensor:
    """Input contribution ``B u`` of a linear model (zero without inputs)."""

    if u is None or model.mat_u is None:
        return torch.zeros_like(like)
    return _mv(model.mat_u, u)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def _filter_op(ei: Elements, ej: Elements) -> Elements:
    """Combine filtering elements (A, b, C, eta, J); ``ei`` precedes ``ej``."""

    Ai, bi, Ci, etai, Ji = ei
    Aj, bj, Cj, etaj, Jj = ej
    eye = torch.eye(Ai.shape[-1], dtype=Ai.dtype)
    AjM = torch.linalg.solve(eye + Ci @ Jj, Aj, left=False)        # A_j (I + C_i J_j)^{-1}
    AiN = torch.linalg.solve(eye + Jj @ Ci, Ai.mT, left=False)     # A_i^T (I + J_j C_i)^{-1}
    return (
        AjM @ Ai,
        _mv(AjM, bi + _mv(Ci, etaj)) + bj,
        AjM @ Ci @ Aj.mT + Cj,
        _mv(AiN, etaj - _mv(Jj, bi)) + etai,
        AiN @ Jj @ Ai + Ji,
    )


class ParallelKalmanFilter(KalmanFilter):
    """
    Kalman filter evaluated as a parallel prefix scan over time.  Same inputs and
    outputs as `KalmanFilter.filter`; much faster for long sequences because
    the Python-level loop has ``log2(T)`` iterations instead of ``T``.
    """

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution] | Tuple:

        if params is not None:
            self.model.update(params)

        dyn, obs = self.dynamics, self.observations
        F, Q, H, R = dyn.mat_x, dyn.noise_cov, obs.mat_x, obs.noise_cov
        m0, P0 = init_dist.mean, init_dist.cov
        T, d = len(data), F.shape[-1]

        # One common batch shape (trajectories x parameters); time axis first.
        batch = torch.broadcast_shapes(
            data.batch_shape, F.shape[:-2], Q.shape[:-2], H.shape[:-2], R.shape[:-2], m0.shape[:-1], P0.shape[:-2],
        )
        Z, z = torch.zeros(T, *batch, d, d), torch.zeros(T, *batch, d)
        y = data.y.expand(*batch, T, data.ydim).movedim(-2, 0)
        observed = data.mask.expand(*batch, T).movedim(-1, 0)          # (T, *batch)
        obs_v, obs_M = observed[..., None], observed[..., None, None]
        y = torch.where(observed[..., None], y, 0.0)
        u = None if data.u is None else data.u.expand(*batch, T, data.udim).movedim(-2, 0)
        # Input driving the transition into row t (row 0 has none when y0=True).
        u_dyn = None if u is None else (torch.cat([torch.zeros_like(u[:1]), u[:-1]]) if y0 else u)
        c = _offset(dyn, u_dyn, z)
        e = _offset(obs, u, y)

        # Generic element: the transition x_t | x_{t-1}, conditioned on y_t.
        S = H @ Q @ H.mT + R
        HtSinv = torch.linalg.solve(S, H, left=True).mT                 # H^T S^{-1}
        K = Q @ HtSinv
        IKH = torch.eye(d) - K @ H
        v = y - _mv(H, c) - e
        A = torch.where(obs_M, IKH @ F, F)
        b = c + torch.where(obs_v, _mv(K, v), 0.0)
        C = torch.where(obs_M, IKH @ Q, Q)
        eta = torch.where(obs_v, _mv(F.mT @ HtSinv, v), 0.0)
        J = torch.where(obs_M, F.mT @ HtSinv @ H @ F, 0.0)

        # First element: the prior (predicted to row 0 unless y_0 observes x_0), conditioned on y_0.
        m_pred0 = m0 if y0 else _mv(F, m0) + c[0]
        P_pred0 = P0 if y0 else F @ P0 @ F.mT + Q
        S0 = H @ P_pred0 @ H.mT + R
        K0 = torch.linalg.solve(S0, H @ P_pred0, left=True).mT
        b0 = m_pred0 + torch.where(obs_v[0], _mv(K0, y[0] - _mv(H, m_pred0) - e[0]), 0.0)
        C0 = torch.where(obs_M[0], P_pred0 - K0 @ S0 @ K0.mT, P_pred0)

        elems = (
            torch.cat([Z[:1], A.expand_as(Z)[1:]]),
            torch.cat([b0.expand_as(z[0])[None], b.expand_as(z)[1:]]),
            torch.cat([C0.expand_as(Z[0])[None], C.expand_as(Z)[1:]]),
            torch.cat([z[:1], eta.expand_as(z)[1:]]),
            torch.cat([Z[:1], J.expand_as(Z)[1:]]),
        )
        _, means, covs, _, _ = _scan(elems, _filter_op)

        log_prob = None
        if compute_log_prob:
            m_pred = torch.cat([m_pred0.expand_as(z[0])[None], _mv(F, means[:-1]) + c[1:]])
            P_pred = torch.cat([P_pred0.expand_as(Z[0])[None], F @ covs[:-1] @ F.mT + Q])
            S_t = H @ P_pred @ H.mT + R
            ll = _gaussian_log_prob_from_sqrt(y - _mv(H, m_pred) - e, torch.linalg.cholesky(S_t))
            log_prob = torch.where(observed, ll, 0.0).sum(0)

        self.dist = FilteringDistribution(mean=means[-1], cov=covs[-1])
        if return_history:
            prior = [] if y0 else [init_dist.snapshot()]
            output = prior + [FilteringDistribution(mean=means[t], cov=covs[t]) for t in range(T)]
        else:
            output = self.dist
        return (output, log_prob) if compute_log_prob else output


# ---------------------------------------------------------------------------
# Smoother
# ---------------------------------------------------------------------------

def _smoother_op(ei: Elements, ej: Elements) -> Elements:
    """Combine smoothing elements (E, g, L); ``ei`` precedes ``ej`` in time."""

    Ei, gi, Li = ei
    Ej, gj, Lj = ej
    return Ei @ Ej, _mv(Ei, gj) + gi, Ei @ Lj @ Ei.mT + Li


class ParallelRTSSmoother(RTSSmoother):
    """
    RTS smoother evaluated as a parallel suffix scan.  Same inputs and outputs as
    `RTSSmoother.smooth` (any linear-Gaussian filter history is accepted).
    """

    def smooth(
        self,
        filtered_history: List[FilteringDistribution],
        u: Tensor | None = None,
        return_gains: bool = False,
    ) -> List[FilteringDistribution] | Tuple[List[FilteringDistribution], List[Tensor]]:

        dyn = self.model.dynamics
        F, Q = dyn.mat_x, dyn.noise_cov
        d = F.shape[-1]

        batch = torch.broadcast_shapes(*(h.mean.shape[:-1] for h in filtered_history), F.shape[:-2], Q.shape[:-2])
        m = torch.stack([h.mean.expand(*batch, d) for h in filtered_history])
        P = torch.stack([h.cov.expand(*batch, d, d) for h in filtered_history])
        K1 = m.shape[0]

        u_t = None if u is None else u.movedim(-2, 0)[: K1 - 1]
        c = _offset(dyn, u_t, m[1:])                                     # input into x_{k+1}
        P_pred = F @ P[:-1] @ F.mT + Q
        E = torch.linalg.solve(P_pred, F @ P[:-1], left=True).mT         # P_k F^T P_pred^{-1}
        g = m[:-1] - _mv(E, _mv(F, m[:-1]) + c)
        L = P[:-1] - E @ F @ P[:-1]

        E = torch.cat([E, torch.zeros_like(E[:1])])
        g = torch.cat([g, m[-1:]])
        L = torch.cat([L, P[-1:]])
        _, ms, Ps = _scan((E, g, L), _smoother_op, reverse=True)

        smoothed = [FilteringDistribution(mean=ms[k], cov=Ps[k]) for k in range(K1)]
        return (smoothed, list(E[:-1])) if return_gains else smoothed
