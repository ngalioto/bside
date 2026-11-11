"""Parallel-in-time Kalman filter / RTS smoother and data masks."""

import pytest
import torch

import bside


def _ssm_with_inputs():
    dt = 0.1
    A = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-9.81, 0.0]]) * dt)
    dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(A), bside.Matrix(torch.tensor([[0.0], [dt]]))),
        noise_cov=1e-2 * torch.eye(2),
    )
    observations = bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(torch.tensor([[1.0, 0.0]]))),
                                             noise_cov=torch.tensor([[0.01]]))
    return bside.SSM(xdim=2, ydim=1, dynamics=dynamics, observations=observations)


def _stack(history, batch):
    return (torch.stack([h.mean.expand(*batch, 2) for h in history]),
            torch.stack([h.cov.expand(*batch, 2, 2) for h in history]))


@pytest.mark.parametrize("y0", [False, True])
@pytest.mark.parametrize("batch", [(), (3,)])
def test_parallel_matches_sequential(y0, batch):
    ssm = _ssm_with_inputs()
    init = bside.FilteringDistribution(mean=torch.tensor([1.0, 0.0]), cov=0.1 * torch.eye(2))
    T = 40
    mask = torch.ones(*batch, T, dtype=torch.bool)
    mask[..., 5] = False
    data = bside.Data(y=torch.randn(*batch, T, 1), u=torch.randn(*batch, T, 1), mask=mask)

    h_seq, ll_seq = bside.KalmanFilter(ssm).filter(data, init, y0=y0, return_history=True, compute_log_prob=True)
    h_par, ll_par = bside.ParallelKalmanFilter(ssm).filter(data, init, y0=y0, return_history=True, compute_log_prob=True)
    assert len(h_seq) == len(h_par)
    for a, b in zip(_stack(h_seq, batch), _stack(h_par, batch)):
        assert torch.allclose(a, b, atol=1e-4)
    assert torch.allclose(ll_seq, ll_par, atol=1e-2)

    s_seq = bside.RTSSmoother(ssm).smooth(h_seq, u=data.u)
    s_par = bside.ParallelRTSSmoother(ssm).smooth(h_par, u=data.u)
    for a, b in zip(_stack(s_seq, batch), _stack(s_par, batch)):
        assert torch.allclose(a, b, atol=1e-4)


def test_mask_and_nan_rows_agree(linear_gaussian_ssm, linear_gaussian_data):
    y = linear_gaussian_data["data"].y.clone()
    mask = torch.rand(y.shape[0]) > 0.3
    init = linear_gaussian_data["init_dist"]
    kf = bside.KalmanFilter(linear_gaussian_ssm)

    _, ll_mask = kf.filter(bside.Data(y=y, mask=mask), init, compute_log_prob=True)
    y_nan = y.clone()
    y_nan[~mask] = float('nan')
    _, ll_nan = kf.filter(bside.Data(y=y_nan), init, compute_log_prob=True)
    _, ll_full = kf.filter(bside.Data(y=y), init, compute_log_prob=True)

    assert torch.allclose(ll_mask, ll_nan)
    assert not torch.allclose(ll_mask, ll_full)  # masked rows really are skipped


def test_cholesky_matrix_is_always_positive_definite():
    Sigma = bside.CholeskyMatrix(torch.eye(3), mask=torch.tril(torch.ones(3, 3, dtype=torch.bool)),
                                 indices=torch.arange(6))
    for _ in range(5):
        Sigma.update(torch.randn(6))
        assert torch.linalg.eigvalsh(Sigma.val).min() > 0
        assert torch.allclose(Sigma.sqrt @ Sigma.sqrt.mT, Sigma.val, atol=1e-5)
    assert torch.allclose(bside.CholeskyMatrix(2 * torch.eye(2)).val, 2 * torch.eye(2))
