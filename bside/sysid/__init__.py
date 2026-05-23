"""
System-identification utilities built on top of bside's filters and smoothers.

* `Posterior` -- a callable adapter that wraps a filter into a log-posterior
  ``log p(theta | data) = log p(theta) + log p(data | theta)`` compatible with
  the `mcmc_samplers.Sampler` ``target`` interface.
* `MultiShootingLoss` -- decoupled multiple-shooting loss for use with any
  ``torch.optim`` optimizer.
* `EM` -- expectation-maximization for linear-Gaussian state-space models
  (Shumway & Stoffer, 1982) with closed-form M-steps for A, C, Q, R, x_0, P_0.
"""

from .posterior import Posterior
from .multi_shooting import MultiShootingLoss
from .em import EM

__all__ = [
    "EM",
    "MultiShootingLoss",
    "Posterior",
]
