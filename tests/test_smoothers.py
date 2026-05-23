"""Smoother sanity tests."""

import torch

import bside
from bside.filtering.smoothers import RTSSmoother, UnscentedRTSSmoother


def test_rts_smoother_reduces_uncertainty(linear_gaussian_ssm, linear_gaussian_data):
    """The smoothed covariance should be (entrywise) <= the filtered covariance."""

    kf = bside.KalmanFilter(model=linear_gaussian_ssm)
    filt_hist = kf.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        return_history=True,
    )

    rts = RTSSmoother(model=linear_gaussian_ssm)
    smooth_hist = rts.smooth(filt_hist)

    for f, s in zip(filt_hist, smooth_hist):
        # tr(P_s) <= tr(P_f) (loose, but should always hold for RTS on the truth model)
        assert torch.trace(s.cov) <= torch.trace(f.cov) + 1e-6


def test_rts_smoother_endpoint_matches_filter(linear_gaussian_ssm, linear_gaussian_data):
    """At the final time step the smoother == filter."""

    kf = bside.KalmanFilter(model=linear_gaussian_ssm)
    filt_hist = kf.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        return_history=True,
    )

    rts = RTSSmoother(model=linear_gaussian_ssm)
    smooth_hist = rts.smooth(filt_hist)

    assert torch.allclose(filt_hist[-1].mean, smooth_hist[-1].mean)
    assert torch.allclose(filt_hist[-1].cov, smooth_hist[-1].cov)


def test_urts_matches_rts_on_linear_system(linear_gaussian_ssm, linear_gaussian_data):
    """Unscented RTS must reproduce linear RTS on a linear system."""

    kf = bside.KalmanFilter(model=linear_gaussian_ssm)
    filt_hist = kf.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        return_history=True,
    )

    rts = RTSSmoother(model=linear_gaussian_ssm)
    rts_hist = rts.smooth(filt_hist)

    urts = UnscentedRTSSmoother(model=linear_gaussian_ssm)
    urts_hist = urts.smooth(filt_hist)

    for a, b in zip(rts_hist, urts_hist):
        assert torch.allclose(a.mean, b.mean, atol=1e-3, rtol=1e-3)
        assert torch.allclose(a.cov, b.cov, atol=1e-3, rtol=1e-3)


def test_particle_smoother_runs_end_to_end(linear_gaussian_ssm, linear_gaussian_data):
    """Forward-filter / backward-simulation smoke test (correctness checked in test_filters)."""

    from bside.filtering.smoothers import ParticleSmoother

    pf = bside.ParticleFilter(model=linear_gaussian_ssm, n_particles=500)
    filt_hist = pf.filter(
        data=linear_gaussian_data["data"],
        init_dist=linear_gaussian_data["init_dist"],
        return_history=True,
    )

    ps = ParticleSmoother(model=linear_gaussian_ssm, n_samples=200)
    smooth_hist = ps.smooth(filt_hist)
    assert len(smooth_hist) == len(filt_hist)
    # Smoothed mean should be finite
    for d in smooth_hist:
        assert torch.isfinite(d.mean).all()
