"""System-identification tests: Posterior, EM, and the mcmc-samplers handshake."""

import importlib

import pytest
import torch

import bside
from bside.sysid import EM, Posterior


def test_posterior_returns_zero_dim_tensor(linear_gaussian_ssm, linear_gaussian_data):
    """Sampler.target signature: scalar tensor in, scalar tensor out."""

    kf = bside.KalmanFilter(model=linear_gaussian_ssm)
    posterior = Posterior(
        filter=kf,
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
    )
    # The KF here doesn't expose parameters because A/C have no mask -- pass
    # anything that update() will silently ignore.
    val = posterior(torch.zeros(0))
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0
    assert torch.isfinite(val)


def test_posterior_with_learnable_dynamics_decreases_on_perturbation(linear_gaussian_data):
    """log p(theta | data) should drop when we perturb theta away from truth."""

    # Build a Kalman filter with a learnable A matrix (full 2x2 mask) on the
    # same physical system as the linear_gaussian fixture.
    dt = 0.1
    g = 9.81
    A_true = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt)
    A = bside.Matrix(
        default=A_true,
        mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(4),
    )
    Q = bside.PSDMatrix(0.01 * torch.eye(2))
    C = bside.Matrix(torch.tensor([[1.0, 0.0]]))
    R = bside.PSDMatrix(torch.tensor([[0.01]]))
    dynamics = bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=Q)
    observations = bside.LinearGaussianModel(model=bside.LinearModel(C), noise_cov=R)
    ssm = bside.SSM(xdim=2, ydim=1, dynamics=dynamics, observations=observations)

    kf = bside.KalmanFilter(model=ssm)
    posterior = Posterior(
        filter=kf,
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
    )

    theta_true = A_true.flatten().clone()
    lp_true = posterior(theta_true)
    lp_perturbed = posterior(theta_true + 0.5 * torch.randn(4))
    assert lp_perturbed < lp_true


def test_em_recovers_linear_gaussian_parameters():
    """
    Generate data from a known linear-Gaussian system, fit EM starting from
    a perturbed init, and check that the recovered A is closer to truth than
    the starting guess.
    """

    A_true = torch.tensor([[0.95, 0.1], [-0.05, 0.9]])
    Q_true = 0.01 * torch.eye(2)
    C_true = torch.eye(2)
    R_true = 0.01 * torch.eye(2)

    true_dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(A_true)),
        noise_cov=bside.PSDMatrix(Q_true),
    )
    true_obs = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(C_true)),
        noise_cov=bside.PSDMatrix(R_true),
    )
    true_ssm = bside.SSM(xdim=2, ydim=2, dynamics=true_dynamics, observations=true_obs)
    x0 = torch.tensor([1.0, -1.0])
    T = 200
    _, y = true_ssm.measure(x=x0, T=T, keep_y0=True, return_x=True)
    data = bside.Data(y=y)

    # Build a separate, mutable SSM to fit
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

    em = EM(ssm=fit_ssm, init_dist=init_dist, y0=True)
    log_marg = em.fit(data, n_iter=40, tol=1e-5)

    A_hat = fit_ssm.dynamics.mat_x
    err_init = (A_init - A_true).abs().max()
    err_final = (A_hat - A_true).abs().max()
    assert err_final < err_init  # EM moved A toward the truth
    # log-likelihood non-decreasing (monotone EM)
    diffs = log_marg[1:] - log_marg[:-1]
    assert (diffs >= -1e-3).all()


@pytest.mark.skipif(
    importlib.util.find_spec("mcmc_samplers") is None,
    reason="mcmc_samplers package not installed",
)
def test_posterior_drives_dram_sampler(linear_gaussian_data):
    """Smoke test: the Posterior is a drop-in target for DRAM."""

    from mcmc_samplers import DelayedRejectionAdaptiveMetropolis

    dt = 0.1
    g = 9.81
    A_true = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt)
    A = bside.Matrix(
        default=A_true,
        mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(4),
    )
    Q = bside.PSDMatrix(0.01 * torch.eye(2))
    C = bside.Matrix(torch.tensor([[1.0, 0.0]]))
    R = bside.PSDMatrix(torch.tensor([[0.01]]))
    dynamics = bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=Q)
    observations = bside.LinearGaussianModel(model=bside.LinearModel(C), noise_cov=R)
    ssm = bside.SSM(xdim=2, ydim=1, dynamics=dynamics, observations=observations)

    kf = bside.KalmanFilter(model=ssm)
    posterior = Posterior(
        filter=kf,
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
    )

    theta0 = A_true.flatten().clone() + 0.01 * torch.randn(4)
    dram = DelayedRejectionAdaptiveMetropolis(
        target=posterior,
        x0=theta0,
        cov=0.001 * torch.eye(4),
    )
    samples, log_probs = dram(N=50, burn_in=10, show_progress=False)
    assert samples.shape == (40, 4)
    assert torch.isfinite(log_probs).all()
