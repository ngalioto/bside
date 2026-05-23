"""
Bayesian parameter inference on a linear pendulum.

Generates a 2-D damped-oscillator trajectory and uses DRAM (delayed-rejection
adaptive Metropolis) from the upstream ``mcmc-samplers`` package to draw samples
from the posterior over the discrete-time dynamics matrix ``A``.  The posterior
target is built by `bside.Posterior` which wraps a `KalmanFilter`'s
``nlog_marginal_likelihood`` together with an optional prior.

Run from the repo root (requires ``pip install mcmc-samplers``)::

    python examples/linear_pendulum.py
"""

import torch
import matplotlib.pyplot as plt

import bside
from mcmc_samplers import DelayedRejectionAdaptiveMetropolis, SampleVisualizer


def main() -> None:
    dt = 0.1
    qc = 0.1
    g = 9.81
    T = 50
    xdim, ydim = 2, 2
    r = 0.1
    measure_y0 = False

    # ---- True system (used only to generate data) ----
    x0 = torch.tensor([1.5, 0.0])
    A_continuous = torch.tensor([[0.0, 1.0], [-g, 0.0]])
    A = bside.Matrix(torch.linalg.matrix_exp(A_continuous * dt))
    C = bside.Matrix(torch.eye(2))
    Q = bside.PSDMatrix(torch.tensor([
        [qc * dt**3 / 3, qc * dt**2 / 2],
        [qc * dt**2 / 2, qc * dt],
    ]))
    R = bside.PSDMatrix(torch.eye(2) * r**2)

    true_dynamics = bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=Q)
    observation_model = bside.LinearGaussianModel(model=bside.LinearModel(C), noise_cov=R)
    true_sys = bside.SSM(xdim=xdim, ydim=ydim, dynamics=true_dynamics, observations=observation_model)
    _, y = true_sys.measure(x=x0, T=T, keep_y0=measure_y0, return_x=True)
    data = bside.Data(y=y)

    # ---- Inference SSM with a learnable A matrix ----
    A_model = bside.Matrix(
        default=torch.eye(xdim),
        mask=torch.ones(xdim, xdim, dtype=bool),
        indices=torch.arange(xdim ** 2),
    )
    learnable_dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(A_model),
        noise_cov=Q,
    )
    sys_model = bside.SSM(
        xdim=xdim, ydim=ydim,
        dynamics=learnable_dynamics,
        observations=observation_model,
    )

    P0 = bside.PSDMatrix(0.01 * torch.eye(xdim))
    init_dist = bside.FilteringDistribution(mean=x0, cov=P0)

    filter = bside.KalmanFilter(model=sys_model)
    posterior = bside.Posterior(
        filter=filter,
        data=data,
        init_dist=init_dist,
        y0=measure_y0,
    )

    init_sample = A_model.val.detach().flatten()
    init_cov = torch.eye(A_model.params.numel()) * 1e-3

    dram = DelayedRejectionAdaptiveMetropolis(
        target=posterior,
        x0=init_sample,
        cov=init_cov,
    )

    with torch.no_grad():
        samples, log_probs = dram(int(1e4))

    print(f"Sampling acceptance rate: {100 * dram.acceptance_ratio:.2f}%")

    visualizer = SampleVisualizer(samples)
    visualizer.triangular_hist(bins=50)
    visualizer.chains()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
