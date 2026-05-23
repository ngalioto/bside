"""Sanity tests for the Matrix / PSDMatrix / DiagonalMatrix / wrapper helpers."""

import pytest
import torch

import bside


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

def test_matrix_round_trip_through_mask():
    default = torch.eye(3)
    mask = torch.tensor([[True, False, True], [False, True, False], [True, False, True]])
    indices = torch.arange(int(mask.sum()))
    M = bside.Matrix(default=default, mask=mask, indices=indices)

    assert M.pdim == int(mask.sum())
    # initial val recovers the default
    assert torch.allclose(M.val, default)

    new_params = torch.arange(int(mask.sum()), dtype=torch.float32) + 10.0
    M.update(new_params)
    assert torch.allclose(M.val[mask], new_params[indices])
    # Off-mask entries stayed at the default values
    assert torch.allclose(M.val[~mask], default[~mask])


def test_matrix_no_mask_update_is_no_op():
    M = bside.Matrix(torch.eye(2))
    M.update(torch.tensor([5.0, 6.0]))  # mask is None -> ignored
    assert torch.allclose(M.val, torch.eye(2))


def test_squared_matrix_squares_params():
    default = torch.zeros(2, 2)
    mask = torch.eye(2, dtype=torch.bool)
    indices = torch.arange(2)
    M = bside.SquaredMatrix(default=default, mask=mask, indices=indices)
    M.update(torch.tensor([3.0, -4.0]))
    diag = torch.diagonal(M.val)
    assert torch.allclose(diag, torch.tensor([9.0, 16.0]))


def test_exponential_matrix_applies_matrix_exp():
    default = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    M = bside.ExponentialMatrix(default=default)
    M.update()
    expected = torch.linalg.matrix_exp(default)
    assert torch.allclose(M.val, expected)


# ---------------------------------------------------------------------------
# PSDMatrix
# ---------------------------------------------------------------------------

def test_psdmatrix_dense_and_sqrt_consistency():
    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    M = bside.PSDMatrix(A)
    assert torch.allclose(M.val, A)
    # Cholesky factor recovers the matrix
    L = M.sqrt
    assert torch.allclose(L @ L.T, A, atol=1e-6)
    # Inverse via Cholesky inverse matches torch.linalg.inv
    assert torch.allclose(M.inv, torch.linalg.inv(A), atol=1e-5)


def test_psdmatrix_sqrt_only_construction():
    L = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
    M = bside.PSDMatrix(default_sqrt=L)
    assert torch.allclose(M.sqrt, L)
    assert torch.allclose(M.val, L @ L.T)


def test_psdmatrix_rejects_upper_triangular_sqrt():
    L = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError):
        bside.PSDMatrix(default_sqrt=L)


# ---------------------------------------------------------------------------
# DiagonalMatrix
# ---------------------------------------------------------------------------

def test_diagonal_matrix_sqrt_and_inv_are_diagonal_only():
    D = torch.diag(torch.tensor([4.0, 9.0, 16.0]))
    M = bside.DiagonalMatrix(D)
    sqrt = M.sqrt
    inv = M.inv
    assert torch.allclose(torch.diagonal(sqrt), torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(torch.diagonal(inv), torch.tensor([0.25, 1.0 / 9.0, 1.0 / 16.0]))
    # Off diagonals are zero
    assert torch.allclose(sqrt - torch.diag(torch.diagonal(sqrt)), torch.zeros_like(sqrt))
    assert torch.allclose(inv - torch.diag(torch.diagonal(inv)), torch.zeros_like(inv))


def test_diagonal_matrix_rejects_non_diagonal_default():
    with pytest.raises(ValueError):
        bside.DiagonalMatrix(torch.tensor([[1.0, 0.5], [0.5, 1.0]]))


# ---------------------------------------------------------------------------
# IdentityModel / LinearModel / AdditiveModel composition
# ---------------------------------------------------------------------------

def test_identity_model_returns_input():
    m = bside.IdentityModel(dim=3)
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(m(x), x)


def test_additive_model_sample_uses_independent_noise_per_particle():
    A = bside.Matrix(torch.eye(2))
    lm = bside.LinearModel(A)
    lg = bside.LinearGaussianModel(model=lm, noise_cov=bside.PSDMatrix(torch.eye(2)))
    particles = torch.zeros(5, 2)
    samples = lg.sample(particles)
    # Each row should be different (with overwhelming probability) because we
    # draw a fresh noise sample per particle.
    assert not torch.allclose(samples[0], samples[1])
    # Mean of many samples ~ 0
    big = lg.sample(torch.zeros(10_000, 2))
    assert torch.allclose(big.mean(dim=0), torch.zeros(2), atol=0.1)


def test_linear_gaussian_model_parameters_are_discoverable():
    """Parameter registration must work after the refactor (no __dict__.update trick)."""
    A_mat = bside.Matrix(
        default=torch.eye(2),
        mask=torch.ones(2, 2, dtype=torch.bool),
        indices=torch.arange(4),
    )
    Q = bside.PSDMatrix(torch.eye(2))
    lg = bside.LinearGaussianModel(model=bside.LinearModel(A_mat), noise_cov=Q)
    params = list(lg.parameters())
    assert len(params) == 1  # only A_mat has learnable params
    assert params[0] is A_mat.params
