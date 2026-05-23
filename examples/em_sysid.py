"""
Expectation-maximization system identification on a linear-Gaussian SSM.

Generates a trajectory from a known linear oscillator, then fits an SSM with
a deliberately wrong initial A matrix using bside's closed-form EM
(Shumway & Stoffer 1982).  The recovered A should converge to the truth and
the marginal log-likelihood should be (almost) monotone non-decreasing.

Run from the repo root::

    python examples/em_sysid.py
"""

import torch
import matplotlib.pyplot as plt

import bside


def main() -> None:
    dt = 0.1
    g = 9.81
    qc = 0.1
    r = 0.1
    T = 300

    A_true = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt)
    Q_true = torch.tensor([
        [qc * dt**3 / 3, qc * dt**2 / 2],
        [qc * dt**2 / 2, qc * dt],
    ])
    C_true = torch.eye(2)
    R_true = (r ** 2) * torch.eye(2)

    # ---- True SSM (used only to generate data) ----
    true_dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(A_true)),
        noise_cov=bside.PSDMatrix(Q_true),
    )
    true_obs = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(C_true)),
        noise_cov=bside.PSDMatrix(R_true),
    )
    true_ssm = bside.SSM(xdim=2, ydim=2, dynamics=true_dynamics, observations=true_obs)
    x0 = torch.tensor([1.5, 0.0])
    _, y = true_ssm.measure(x=x0, T=T, keep_y0=True, return_x=True)
    data = bside.Data(y=y)

    # ---- SSM to fit (start far from truth) ----
    A_init = torch.eye(2)
    fit_dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(A_init)),
        noise_cov=bside.PSDMatrix(0.1 * torch.eye(2)),
    )
    fit_obs = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(C_true.clone())),
        noise_cov=bside.PSDMatrix(0.1 * torch.eye(2)),
    )
    fit_ssm = bside.SSM(xdim=2, ydim=2, dynamics=fit_dynamics, observations=fit_obs)
    init_dist = bside.FilteringDistribution(mean=x0, cov=0.01 * torch.eye(2))

    em = bside.EM(ssm=fit_ssm, init_dist=init_dist, y0=True)
    log_marg = em.fit(data, n_iter=80, tol=1e-7, verbose=True)

    A_hat = fit_ssm.dynamics.mat_x
    print("\nA_true:\n", A_true)
    print("A_init:\n", A_init)
    print("A_hat:\n", A_hat)
    print(f"\n||A_hat - A_true||_F = {(A_hat - A_true).norm():.4e}")

    plt.figure(figsize=(8, 4))
    plt.plot(log_marg, marker="o")
    plt.xlabel("EM iteration")
    plt.ylabel("log p(y) (filter marginal)")
    plt.title("EM convergence on linear-Gaussian SSM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
