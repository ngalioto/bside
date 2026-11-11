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

from typing import Callable

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
        """Log marginal likelihood ``log p(y | theta)``."""

        return -self.filter.nlog_marginal_likelihood(
            data=self.data,
            init_dist=self.init_dist,
            y0=self.y0,
            params=params,
        )

    def log_prior(
        self,
        params: Tensor,
    ) -> Tensor:
        if self.prior is None:
            return torch.zeros((), dtype=params.dtype, device=params.device)
        return self.prior(params)

    def __call__(
        self,
        params: Tensor,
    ) -> Tensor:
        # Samplers may pass a single point as a (1, p) row; the model expects (p,)
        # and the sampler expects the log-density back with the same leading shape.
        lead = params.shape[:-1]
        theta = params.reshape(-1)
        log_prior = self.log_prior(theta)
        if not torch.isfinite(log_prior):
            # Outside the prior's support: skip the filter (the model may be invalid there).
            return log_prior.reshape(lead)
        return (self.log_likelihood(theta) + log_prior).reshape(lead)
