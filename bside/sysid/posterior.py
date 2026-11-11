"""
Posterior adapter that wraps a `Filter` into a single-arg callable suitable for
use as ``target`` in `mcmc_samplers.Sampler` subclasses.

Usage::

    from bside import KalmanFilter, FilteringDistribution
    from bside.sysid import Posterior
    from mcmc_samplers import DelayedRejectionAdaptiveMetropolis

    filter = KalmanFilter(model=ssm)
    posterior = Posterior(filter, data, init_dist, prior=my_log_prior)
    dram = DelayedRejectionAdaptiveMetropolis(target=posterior, x0=theta0, cov=cov0)
    samples, log_probs = dram(N=10_000)

The posterior is ``log p(y_{1:T} | theta) + log p(theta)``.  When the prior is
omitted an improper flat prior is used.  The filter's parameter-update hook
(``self.model.update(params)``) is invoked once per evaluation so it works
transparently for both the standard `Filter` family and `ParticleFilter`.
"""

from typing import Callable, Tuple

import torch
from torch import Tensor

from bside.dataset import Data
from bside.filtering import FilteringDistribution


class Posterior:
    """
    Callable target ``params -> log p(theta) + log p(y | theta)`` driven by an
    underlying filter's marginal-likelihood routine.
    """

    def __init__(
        self,
        filter,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        prior: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        filter
            Any object exposing ``nlog_marginal_likelihood(data, init_dist, y0, params)``.
            Concretely, every `Filter` subclass shipped with bside, plus
            `ParticleFilter`.
        data : Data
            Observation sequence (and optional control inputs).
        init_dist : FilteringDistribution
            Prior on the initial state.
        y0 : bool, optional
            Whether to condition on the very first observation. See
            `Filter.filter`.
        prior : callable, optional
            ``prior(params) -> log p(params)``. Omit for an improper flat prior.
        """

        self.filter = filter
        self.data = data
        self.init_dist = init_dist
        self.y0 = y0
        self.prior = prior

    def log_likelihood(
        self,
        params: Tensor,
    ) -> Tensor:
        """
        Log marginal likelihood ``log p(y | theta)`` for ``params`` of shape
        ``(..., p)``; returns shape ``(...)``.  Independent trajectories in a
        batched ``data`` are summed.

        A batch of parameter vectors is evaluated in one filter pass when the
        filter supports batching (all Gaussian filters): the model's matrices
        and maps receive the batch and broadcast it against the trajectories.
        Other filters (particle, ensemble) loop over the batch.
        """

        lead = params.shape[:-1]
        if not lead:
            return -self.filter.nlog_marginal_likelihood(
                data=self.data, init_dist=self.init_dist, y0=self.y0, params=params,
            )
        if not self.filter.supports_batch:
            flat = params.reshape(-1, params.shape[-1])
            return torch.stack([self.log_likelihood(theta) for theta in flat]).reshape(lead)

        # Parameter batch dims lead, trajectory batch dims follow.
        n_traj = len(self.data.batch_shape)
        theta = params.reshape(*lead, *([1] * n_traj), params.shape[-1])
        # Give the prior the parameter batch dims too, so the first predict / update
        # (e.g. sigma points) already lines up with the batched model.
        ones = [1] * n_traj
        m0, P0 = self.init_dist.mean, self.init_dist.cov
        init_dist = FilteringDistribution(
            mean=m0.expand(*lead, *ones, m0.shape[-1]),
            cov=P0.expand(*lead, *ones, *P0.shape[-2:]),
        )
        _, log_prob = self.filter.filter(
            data=self.data, init_dist=init_dist, y0=self.y0, compute_log_prob=True, params=theta,
        )
        log_prob = torch.as_tensor(log_prob).expand(*lead, *self.data.batch_shape)
        return log_prob.sum(dim=tuple(range(len(lead), log_prob.ndim))) if n_traj else log_prob

    def log_prior(
        self,
        params: Tensor,
    ) -> Tensor:
        """``log p(theta)``; a custom prior must map ``(..., p)`` to ``(...)``."""

        if self.prior is None:
            return torch.zeros(params.shape[:-1], dtype=params.dtype, device=params.device)
        return torch.as_tensor(self.prior(params), dtype=params.dtype).expand(params.shape[:-1])

    def __call__(
        self,
        params: Tensor,
    ) -> Tensor:
        """
        ``log p(theta) + log p(y | theta)`` for ``params`` of shape ``(..., p)``.
        Where the prior is zero the filter is not run (the model may be invalid
        there) and ``-inf`` is returned.  Differentiable for the Gaussian filters.
        """

        log_prior = self.log_prior(params)
        valid = torch.isfinite(log_prior)
        if not valid.any():
            return log_prior
        if not valid.all():
            # Evaluate invalid rows at a valid point, then mask them out.
            fill = params[valid][0]
            params = torch.where(valid.unsqueeze(-1), params, fill.detach())
        log_post = self.log_likelihood(params) + log_prior
        return torch.where(valid, log_post, log_prior)

    def log_prob_and_grad(
        self,
        params: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Log posterior and its gradient w.r.t. ``params`` (row-wise for a batch)."""

        with torch.enable_grad():
            theta = params.detach().requires_grad_(True)
            log_post = self(theta)
            finite = torch.isfinite(log_post)
            (grad,) = torch.autograd.grad(log_post[finite].sum(), theta, allow_unused=True) if finite.any() else (None,)
        grad = torch.zeros_like(theta) if grad is None else grad
        return log_post.detach(), grad
