"""Batched / differentiable posterior, MAP, HMC and process-noise continuation."""

import torch

import bside


def _learnable_A_posterior(linear_gaussian_ssm, linear_gaussian_data, filter_cls=bside.KalmanFilter, **kw):
    A_true = linear_gaussian_ssm.dynamics.mat_x
    A = bside.Matrix(A_true.clone(), mask=torch.ones(2, 2, dtype=torch.bool), indices=torch.arange(4))
    ssm = bside.SSM(xdim=2, ydim=1,
                    dynamics=bside.LinearGaussianModel(model=bside.LinearModel(A),
                                                       noise_cov=linear_gaussian_ssm.dynamics.noise_cov),
                    observations=linear_gaussian_ssm.observations)
    flt = filter_cls(ssm, **kw)
    post = bside.Posterior(flt, linear_gaussian_data["data"], linear_gaussian_data["init_dist"])
    return post, A_true.flatten()


def test_batched_parameters_match_loop(linear_gaussian_ssm, linear_gaussian_data):
    for cls, kw in [(bside.KalmanFilter, {}), (bside.UnscentedKalmanFilter, {})]:
        post, theta = _learnable_A_posterior(linear_gaussian_ssm, linear_gaussian_data, cls, **kw)
        thetas = theta + 0.01 * torch.randn(4, 4)
        batched = post(thetas)
        looped = torch.stack([post(t) for t in thetas])
        assert batched.shape == (4,)
        assert torch.allclose(batched, looped, rtol=1e-4, atol=1e-2)


def test_gradient_matches_finite_difference():
    """Autograd through the Kalman filter agrees with central differences (float64)."""

    torch.set_default_dtype(torch.float64)
    try:
        A = bside.Matrix(torch.tensor([[0.9, 0.1], [-0.2, 0.9]]), mask=torch.ones(2, 2, dtype=torch.bool),
                         indices=torch.arange(4))
        ssm = bside.SSM(xdim=2, ydim=1,
                        dynamics=bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=0.01 * torch.eye(2)),
                        observations=bside.LinearGaussianModel(
                            model=bside.LinearModel(bside.Matrix(torch.tensor([[1.0, 0.0]]))),
                            noise_cov=torch.tensor([[0.01]])))
        data = bside.Data(y=torch.randn(30, 1))
        post = bside.Posterior(bside.KalmanFilter(ssm), data,
                               bside.FilteringDistribution(mean=torch.zeros(2), cov=torch.eye(2)))
        theta = torch.tensor([0.9, 0.1, -0.2, 0.9])
        _, grad = post.log_prob_and_grad(theta)
        h = 1e-6
        fd = torch.stack([(post(theta + h * e) - post(theta - h * e)) / (2 * h) for e in torch.eye(4)])
        assert torch.allclose(grad, fd, rtol=1e-5, atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float32)


def _gaussian_target(mean, std):
    return lambda x: (-0.5 * ((x - mean) / std) ** 2).sum(-1)


def test_map_estimate_multistart():
    target = _gaussian_target(torch.tensor([1.0, -2.0]), torch.tensor([1.0, 0.5]))
    theta, trace = bside.map_estimate(target, torch.randn(5, 2), n_steps=300, lr=0.1)
    assert torch.allclose(theta, torch.tensor([1.0, -2.0]).expand(5, 2), atol=1e-2)
    assert trace.shape == (300, 5)


def test_hmc_recovers_gaussian_moments():
    mean, std = torch.tensor([1.0, -2.0]), torch.tensor([1.0, 0.1])
    hmc = bside.HMC(_gaussian_target(mean, std), n_leapfrog=10)
    samples, _ = hmc(torch.zeros(8, 2), n_samples=500, n_warmup=300)
    flat = samples.reshape(-1, 2)
    assert torch.allclose(flat.mean(0), mean, atol=0.1)
    assert torch.allclose(flat.std(0), std, rtol=0.15)
    assert 0.5 < hmc.acceptance_rate <= 1.0


def test_noise_continuation_profiles_and_picks_best_level():
    # theta = (a, log_s): the best fit is a = 3 at log_s = -1.
    target = lambda x: -((x[..., 0] - 3.0) ** 2) - (x[..., 1] + 1.0) ** 2  # noqa: E731
    prof = bside.noise_continuation(target, torch.zeros(2), noise_indices=[1], levels=[1.0, 0.0, -1.0, -2.0],
                                    n_steps=200, lr=0.1)
    assert prof.log_posteriors.shape == (4,)
    assert prof.best == 2
    assert torch.allclose(prof.thetas[:, 0], torch.full((4,), 3.0), atol=1e-2)
    assert torch.allclose(prof.theta, torch.tensor([3.0, -1.0]), atol=1e-2)
