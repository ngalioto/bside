"""
Unscented Kalman filter on the Lorenz '63 system.

Demonstrates state estimation for a chaotic nonlinear system observed only
through its first coordinate.  The UKF tracks the unobserved (y, z) components
through the cross-coupling in the dynamics.

Run from the repo root::

    python examples/lorenz63_ukf.py
"""

import torch
import matplotlib.pyplot as plt

import bside


def lorenz_step(x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    """Single explicit-Euler step of the Lorenz '63 attractor."""
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
    T = 500

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

    # Prior on the initial state: rough, off-center, with non-trivial covariance.
    init_dist = bside.FilteringDistribution(
        mean=x0 + 2.0 * torch.randn(xdim),
        cov=4.0 * torch.eye(xdim),
    )

    ukf = bside.UnscentedKalmanFilter(model=ssm)
    hist = ukf.filter(data=data, init_dist=init_dist, y0=True, return_history=True)

    means = torch.stack([d.mean for d in hist])
    stds = torch.stack([torch.sqrt(torch.diagonal(d.cov)) for d in hist])

    t = torch.arange(T + 1) * 0.01
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    labels = ["x", "y", "z"]
    for ii, ax in enumerate(axes):
        ax.plot(t, x_true[:, ii], "k", label="true")
        ax.plot(t, means[:, ii], "C0", label="UKF mean")
        ax.fill_between(
            t,
            means[:, ii] - 2 * stds[:, ii],
            means[:, ii] + 2 * stds[:, ii],
            color="C0", alpha=0.15,
        )
        ax.set_ylabel(labels[ii])
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.suptitle("UKF on Lorenz '63 (observing x only)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
