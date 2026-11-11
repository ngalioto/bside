"""
Gradient-based inference on a filter-based posterior.

* `map_estimate` -- maximize ``log p(theta | y)`` with a torch optimizer.  A
  batch of starting points ``(P, p)`` is optimized in parallel (multi-start).
* `HMC` -- Hamiltonian Monte Carlo with step-size (dual averaging) and diagonal
  mass-matrix adaptation during warm-up; a batch of starting points runs
  independent chains in parallel.
* `noise_continuation` -- fit the dynamics while stepping the process-noise
  level down from large to small, recording the profile posterior at every
  level.

All of them need a differentiable target such as `Posterior` with a Gaussian
filter (the particle filter's resampling is not differentiable).
"""

from dataclasses import dataclass
from math import log
from typing import Callable, Sequence

import torch
from torch import Tensor

Target = Callable[[Tensor], Tensor]


def _log_prob_and_grad(target: Target, theta: Tensor):
    if hasattr(target, 'log_prob_and_grad'):
        return target.log_prob_and_grad(theta)
    with torch.enable_grad():
        x = theta.detach().requires_grad_(True)
        lp = target(x)
        finite = torch.isfinite(lp)
        grad = torch.autograd.grad(lp[finite].sum(), x)[0] if finite.any() else torch.zeros_like(x)
    return lp.detach(), grad


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------

def map_estimate(
    target: Target,
    theta0: Tensor,
    n_steps: int = 500,
    lr: float = 0.05,
    fixed: Tensor | Sequence[int] | None = None,
    optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
) -> tuple[Tensor, Tensor]:
    """
    Maximize ``target(theta)`` (a log posterior) by gradient ascent.

    Parameters
    ----------
    theta0 : Tensor
        Starting point ``(p,)``, or a batch ``(P, p)`` optimized in parallel.
    fixed : indices, optional
        Entries of ``theta`` held at their starting values.

    Returns
    -------
    theta : Tensor
        The optimum (same shape as ``theta0``).
    trace : Tensor ``(n_steps, ...)``
        Log posterior at every step.
    """

    theta = theta0.detach().clone().requires_grad_(True)
    free = torch.ones(theta.shape[-1])
    if fixed is not None:
        free[torch.as_tensor(fixed)] = 0.0
    opt = optimizer([theta], lr=lr)

    trace = []
    for _ in range(n_steps):
        opt.zero_grad()
        log_post = target(theta)
        (-log_post.sum()).backward()
        theta.grad.mul_(free).nan_to_num_(0.0)
        opt.step()
        trace.append(log_post.detach())

    return theta.detach(), torch.stack(trace)


# ---------------------------------------------------------------------------
# Process-noise continuation
# ---------------------------------------------------------------------------

@dataclass
class NoiseProfile:
    """Result of `noise_continuation`."""

    levels: Tensor          #: noise levels visited (in order)
    thetas: Tensor          #: optimum at each level, ``(n_levels, p)``
    log_posteriors: Tensor  #: profile log posterior at each level
    theta: Tensor           #: final estimate (joint fit from the best level)

    @property
    def best(self) -> Tensor:
        """Index of the best level (per starting point for a batch of starts)."""
        return torch.argmax(self.log_posteriors, dim=0)


def noise_continuation(
    target: Target,
    theta0: Tensor,
    noise_indices: Sequence[int],
    levels: Sequence[float] | Tensor,
    n_steps: int = 200,
    lr: float = 0.05,
    final_steps: int = 0,
) -> NoiseProfile:
    """
    Learn dynamics and process noise without getting stuck at high noise.

    Maximizing jointly over the dynamics parameters and the process-noise
    covariance ``Sigma`` often stalls in a high-noise local optimum. There the
    filter explains the data as noise, so the likelihood barely depends on the
    dynamics. Large noise is still useful early on: it smooths the likelihood
    landscape (especially for chaotic systems). This routine therefore:

    1. fixes the noise parameters at each value in ``levels`` (pass them from
       large to small), maximizing over the remaining parameters and warm-starting
       from the previous level, so the dynamics are carried into the basin
       of a small-noise fit;
    2. records the profile log posterior ``max_theta log p(theta, Sigma | y)`` at each level, which shows
       the trade-off between process noise and model fit directly;
    3. optionally (``final_steps > 0``) frees all parameters and refines from the
       best level, which starts the joint fit inside the low-noise basin.

    ``noise_indices`` are the entries of ``theta`` that set the noise level, e.g. the
    log-standard-deviation entries of a diagonal `bside.CholeskyMatrix`; each level is
    written to all of them.  A batch of starting points ``(P, p)`` is continued in
    parallel, and the best level is chosen per start.
    """

    idx = torch.as_tensor(noise_indices)
    levels = torch.as_tensor(levels, dtype=theta0.dtype)
    theta = theta0.detach().clone()

    thetas, log_posts = [], []
    for level in levels:
        theta[..., idx] = level
        theta, trace = map_estimate(target, theta, n_steps=n_steps, lr=lr, fixed=idx)
        thetas.append(theta.clone())
        log_posts.append(target(theta).detach())

    thetas, log_posts = torch.stack(thetas), torch.stack(log_posts)          # (L, ..., p), (L, ...)
    # Best level for each starting point.
    best_idx = torch.argmax(log_posts, dim=0)
    best = torch.gather(thetas, 0, best_idx[None, ..., None].expand(1, *thetas.shape[1:]))[0]
    final = map_estimate(target, best, n_steps=final_steps, lr=lr)[0] if final_steps else best
    return NoiseProfile(levels=levels, thetas=thetas, log_posteriors=log_posts, theta=final)


# ---------------------------------------------------------------------------
# Hamiltonian Monte Carlo
# ---------------------------------------------------------------------------

class HMC:
    """
    Hamiltonian Monte Carlo (Neal, 2011) with warm-up adaptation:

    * step size by dual averaging toward ``target_accept`` (Hoffman & Gelman, 2014)
    * a diagonal mass matrix from the sample variance over the middle of warm-up
    * a trajectory length drawn uniformly from ``1..n_leapfrog`` every iteration

    A batch of starting points ``(P, p)`` runs ``P`` independent chains, which is
    nearly free when the target evaluates parameter batches in one pass.
    """

    def __init__(
        self,
        target: Target,
        step_size: float = 0.01,
        n_leapfrog: int = 20,
        target_accept: float = 0.8,
    ) -> None:

        self.target = target
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.target_accept = target_accept
        self.inv_mass = None

    def _leapfrog(self, theta, momentum, grad, eps, n_steps):
        momentum = momentum + 0.5 * eps * grad
        for i in range(n_steps):
            theta = theta + eps * self.inv_mass * momentum
            log_prob, grad = _log_prob_and_grad(self.target, theta)
            if i < n_steps - 1:
                momentum = momentum + eps * grad
        momentum = momentum + 0.5 * eps * grad
        return theta, momentum, log_prob, grad

    def __call__(
        self,
        theta0: Tensor,
        n_samples: int,
        n_warmup: int = 500,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns ``samples`` with shape ``(n_samples, *theta0.shape)`` and the log
        target at each sample; warm-up draws are discarded.
        """

        theta = theta0.detach().clone()
        log_prob, grad = _log_prob_and_grad(self.target, theta)
        self.inv_mass = torch.ones(theta.shape[-1])
        mass_window = (n_warmup // 4, 3 * n_warmup // 4)
        warm = []

        # Dual-averaging state
        eps = self.step_size
        mu, h_bar, log_eps_bar = log(10 * eps), 0.0, 0.0
        gamma, t0, kappa = 0.05, 10.0, 0.75
        n_adapt = 0

        samples, log_probs, accepts = [], [], []
        for it in range(n_warmup + n_samples):
            momentum = torch.randn_like(theta) / self.inv_mass.sqrt()
            kinetic0 = 0.5 * (momentum**2 * self.inv_mass).sum(-1)
            # A random trajectory length avoids near-periodic (non-mixing) trajectories.
            n_steps = int(torch.randint(1, self.n_leapfrog + 1, ()))
            prop, p_new, lp_new, g_new = self._leapfrog(theta, momentum, grad, eps, n_steps)
            kinetic1 = 0.5 * (p_new**2 * self.inv_mass).sum(-1)

            log_alpha = (lp_new - kinetic1) - (log_prob - kinetic0)
            alpha = torch.nan_to_num(log_alpha.clamp(max=0.0).exp(), nan=0.0)
            accept = torch.rand_like(alpha) < alpha
            theta = torch.where(accept.unsqueeze(-1), prop, theta)
            log_prob = torch.where(accept, lp_new, log_prob)
            grad = torch.where(accept.unsqueeze(-1), g_new, grad)

            if it < n_warmup:
                n_adapt += 1
                h_bar = (1 - 1 / (n_adapt + t0)) * h_bar + (self.target_accept - alpha.mean().item()) / (n_adapt + t0)
                log_eps = mu - n_adapt**0.5 / gamma * h_bar
                eta = n_adapt**-kappa
                log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
                eps = float(torch.tensor(log_eps).exp())
                if mass_window[0] <= it < mass_window[1]:
                    warm.append(theta.reshape(-1, theta.shape[-1]))
                if it == mass_window[1] - 1 and warm:
                    # Diagonal mass from the warm-up variance, then restart step-size adaptation.
                    self.inv_mass = torch.cat(warm).var(0).clamp(min=1e-8)
                    mu, h_bar, log_eps_bar, n_adapt = log(10 * eps), 0.0, 0.0, 0
                if it == n_warmup - 1:
                    eps = float(torch.tensor(log_eps_bar).exp())
            else:
                samples.append(theta.clone())
                log_probs.append(log_prob.clone())
                accepts.append(alpha.mean())

        self.step_size = eps
        self.acceptance_rate = torch.stack(accepts).mean().item() if accepts else float('nan')
        return torch.stack(samples), torch.stack(log_probs)
