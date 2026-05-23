"""
Bootstrap particle filter on the Lorenz '63 system.

Identical setup to ``lorenz63_ukf.py`` but with a `ParticleFilter` instead of
a UKF, demonstrating non-Gaussian posterior tracking.  The weighted-particle
mean is plotted together with an ensemble cloud at a few snapshot times.

Run from the repo root::

    python examples/lorenz63_pf.py
"""

import torch
import matplotlib.pyplot as plt

import bside


def lorenz_step(x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
    dx = torch.stack([
        sigma * (x[..., 1] - x[..., 0]),
        x[..., 0] * (rho - x[..., 2]) - x[..., 1],
        x[..., 0] * x[..., 1] - beta * x[..., 2],
    ], dim=-1)
    return x + dt * dx


def observe_x_only(x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    return x[..., 0:1]


def main() -> None:
    xdim, ydim = 3, 1
    T = 400
    n_particles = 2000

    Q = bside.PSDMatrix(1e-4 * torch.eye(xdim))
    R = bside.PSDMatrix(0.5 * torch.eye(ydim))

    dynamics = bside.NonlinearAdditiveModel(
        f=lorenz_step, noise_cov=Q, in_dim=xdim, out_dim=xdim,
    )
    observations = bside.NonlinearAdditiveModel(
        f=observe_x_only, noise_cov=R, in_dim=xdim, out_dim=ydim,
    )
    ssm = bside.SSM(xdim=xdim, ydim=ydim, dynamics=dynamics, observations=observations)

    x0 = torch.tensor([1.0, 1.0, 1.0])
    x_true, y = ssm.measure(x=x0, T=T, keep_y0=True, return_x=True)
    data = bside.Data(y=y)

    init_dist = bside.FilteringDistribution(
        mean=x0 + torch.randn(xdim),
        cov=torch.eye(xdim),
    )

    pf = bside.ParticleFilter(
        model=ssm,
        n_particles=n_particles,
        resample_method="systematic",
        ess_threshold=0.5,
    )
    hist = pf.filter(data=data, init_dist=init_dist, y0=True, return_history=True)

    means = torch.stack([d.weighted_mean() for d in hist])

    t = torch.arange(T + 1) * 0.01
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    labels = ["x", "y", "z"]
    for ii, ax in enumerate(axes):
        ax.plot(t, x_true[:, ii], "k", label="true")
        ax.plot(t, means[:, ii], "C2", label="PF weighted mean")
        ax.set_ylabel(labels[ii])
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.suptitle(f"Particle filter on Lorenz '63, N={n_particles}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
