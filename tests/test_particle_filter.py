"""Particle filter sanity tests."""

import torch

import bside
from bside.filtering import functional as F


def test_particle_filter_tracks_kalman_mean(linear_gaussian_ssm, linear_gaussian_data):
    """
    The bootstrap PF posterior mean must converge to the KF mean as the number
    of particles grows.  We use a generous tolerance to keep the test fast.
    """

    kf = bside.KalmanFilter(model=linear_gaussian_ssm)
    kf_hist = kf.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        return_history=True,
    )
    kf_means = torch.stack([d.mean for d in kf_hist[1:]])  # drop prior

    pf = bside.ParticleFilter(model=linear_gaussian_ssm, n_particles=5_000, ess_threshold=0.5)
    pf_hist = pf.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        return_history=True,
    )
    pf_means = torch.stack([d.weighted_mean() for d in pf_hist[1:]])

    err = (pf_means - kf_means).pow(2).mean().sqrt()
    assert err < 0.25


def test_resampling_preserves_total_weight():
    """After resampling, weights must sum to one and be uniform."""

    n = 100
    particles = torch.randn(n, 3)
    log_w = torch.randn(n)
    dist = bside.FilteringDistribution(particles=particles, log_weights=log_w)
    dist.resample(method='systematic')
    w = dist.normalized_weights()
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    assert torch.allclose(w, torch.full((n,), 1.0 / n), atol=1e-6)


def test_effective_sample_size_uniform_is_n():
    n = 50
    particles = torch.randn(n, 2)
    log_w = torch.full((n,), -torch.log(torch.tensor(float(n))))
    dist = bside.FilteringDistribution(particles=particles, log_weights=log_w)
    assert torch.isclose(dist.effective_sample_size(), torch.tensor(float(n)), atol=1e-3)


def test_pf_log_marginal_likelihood_finite(linear_gaussian_ssm, linear_gaussian_data):
    pf = bside.ParticleFilter(model=linear_gaussian_ssm, n_particles=1000)
    nll = pf.nlog_marginal_likelihood(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
    )
    assert torch.isfinite(torch.as_tensor(nll))
