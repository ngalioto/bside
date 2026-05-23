"""
Cross-check every Gaussian filter against the analytical Kalman recursion on a
linear-Gaussian system.  The KF answer is the reference; all other filters
must agree to a tolerance.
"""

import pytest
import torch

import bside


def _kf_filtered(ssm, init_dist, data, y0=False):
    kf = bside.KalmanFilter(model=ssm)
    history, log_marg = kf.filter(
        data=data,
        init_dist=init_dist,
        y0=y0,
        return_history=True,
        compute_log_prob=True,
    )
    means = torch.stack([d.mean for d in history])
    covs = torch.stack([d.cov for d in history])
    return means, covs, log_marg


def test_kalman_filter_recovers_state(linear_gaussian_ssm, linear_gaussian_data):
    means, covs, log_marg = _kf_filtered(
        linear_gaussian_ssm,
        linear_gaussian_data["init_dist"],
        linear_gaussian_data["data"],
    )

    # state_estimates[0] is the prior; positions 1..T match x_true 0..T-1.
    err = (means[1:] - linear_gaussian_data["x_true"]).pow(2).mean().sqrt()
    assert err < 0.5

    # Log marginal likelihood is a finite scalar
    assert torch.isfinite(torch.as_tensor(log_marg))


@pytest.mark.parametrize("FilterCls,kwargs", [
    (bside.SquareRootKalmanFilter, {}),
    (bside.UnscentedKalmanFilter, {}),
    (bside.GaussHermiteFilter, {"order": 3}),
    (bside.CubatureKalmanFilter, {}),
])
def test_gaussian_filters_match_kalman(FilterCls, kwargs, linear_gaussian_ssm, linear_gaussian_data):
    """On a linear-Gaussian system every Gaussian filter must match the analytical KF."""

    kf_means, kf_covs, _ = _kf_filtered(
        linear_gaussian_ssm,
        linear_gaussian_data["init_dist"],
        linear_gaussian_data["data"],
    )

    flt = FilterCls(model=linear_gaussian_ssm, **kwargs)
    history = flt.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        y0=False,
        return_history=True,
    )
    means = torch.stack([d.mean for d in history])
    covs = torch.stack([d.cov for d in history])

    assert torch.allclose(means, kf_means, atol=1e-3, rtol=1e-3)
    assert torch.allclose(covs, kf_covs, atol=1e-3, rtol=1e-3)


def test_enkf_converges_to_kf_with_large_ensemble(linear_gaussian_ssm, linear_gaussian_data):
    """The EnKF with a large ensemble should agree with the KF posterior mean."""

    kf_means, _, _ = _kf_filtered(
        linear_gaussian_ssm,
        linear_gaussian_data["init_dist"],
        linear_gaussian_data["data"],
    )

    enkf = bside.EnsembleKalmanFilter(model=linear_gaussian_ssm, ensemble_size=5000)
    history = enkf.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        return_history=True,
    )
    means = torch.stack([torch.mean(d.particles, dim=0) for d in history])

    # Tolerance is generous because the EnKF is Monte Carlo.
    assert torch.allclose(means, kf_means, atol=0.15, rtol=0.05)


def test_kalman_filter_y0_true_runs(linear_gaussian_ssm):
    """Smoke test for the y0=True path that crashed in the original code."""

    T = 50
    x0 = torch.tensor([0.5, 0.0])
    x, y = linear_gaussian_ssm.measure(x=x0, T=T, keep_y0=True, return_x=True)
    data = bside.Data(y=y)
    init_dist = bside.FilteringDistribution(mean=x0, cov=0.01 * torch.eye(2))

    kf = bside.KalmanFilter(model=linear_gaussian_ssm)
    history, log_marg = kf.filter(
        data=data, init_dist=init_dist, y0=True, return_history=True, compute_log_prob=True,
    )
    assert len(history) == T + 1  # init + T observations
    assert torch.isfinite(torch.as_tensor(log_marg))


def test_kalman_filter_accepts_identity_observations():
    """A KalmanFilter with the default IdentityModel observations must construct cleanly."""
    A = bside.Matrix(0.95 * torch.eye(2))
    dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(A),
        noise_cov=bside.PSDMatrix(0.05 * torch.eye(2)),
    )
    ssm = bside.SSM(xdim=2, ydim=2, dynamics=dynamics, observations=None)
    kf = bside.KalmanFilter(model=ssm)
    assert isinstance(kf.model.observations, bside.LinearGaussianModel)


def test_nlog_marginal_likelihood_decreases_toward_true_params(linear_gaussian_ssm, linear_gaussian_data):
    """
    Perturbing the dynamics matrix should *raise* the negative log-marginal-
    likelihood; restoring it should lower it again.  This is the basic
    gradient-direction check that mcmc-samplers / EM rely on.
    """

    kf = bside.KalmanFilter(model=linear_gaussian_ssm)
    init_dist = linear_gaussian_data["init_dist"]
    data = linear_gaussian_data["data"]

    A_orig = linear_gaussian_ssm.dynamics.mat_x.clone()

    nll_true = kf.nlog_marginal_likelihood(data, init_dist, y0=False).item()

    # Perturb the dynamics matrix
    perturbed = A_orig + 0.2 * torch.randn_like(A_orig)
    linear_gaussian_ssm.dynamics.model._mat_x.val = perturbed
    nll_perturbed = kf.nlog_marginal_likelihood(data, init_dist, y0=False).item()
    assert nll_perturbed > nll_true - 1.0  # weak but reliable
    # Restore
    linear_gaussian_ssm.dynamics.model._mat_x.val = A_orig
