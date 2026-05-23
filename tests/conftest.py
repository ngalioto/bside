"""Shared pytest fixtures for the bside test suite."""

import pytest
import torch

import bside


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed torch for reproducibility on every test."""
    torch.manual_seed(0)


@pytest.fixture
def linear_gaussian_ssm():
    """
    A 2D damped harmonic-oscillator state-space model with full state
    observations.  Used as the analytical reference for every Gaussian filter
    test.
    """

    dt = 0.1
    g = 9.81
    qc = 1e-2
    r = 1e-1

    xdim, ydim = 2, 1
    A_cont = torch.tensor([[0.0, 1.0], [-g, 0.0]])
    A = bside.Matrix(torch.linalg.matrix_exp(A_cont * dt))
    C = bside.Matrix(torch.tensor([[1.0, 0.0]]))
    Q = bside.PSDMatrix(torch.tensor([
        [qc * dt**3 / 3, qc * dt**2 / 2],
        [qc * dt**2 / 2, qc * dt],
    ]))
    R = bside.PSDMatrix(torch.tensor([[r ** 2]]))

    dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(A),
        noise_cov=Q,
    )
    observations = bside.LinearGaussianModel(
        model=bside.LinearModel(C),
        noise_cov=R,
    )
    return bside.SSM(xdim=xdim, ydim=ydim, dynamics=dynamics, observations=observations)


@pytest.fixture
def linear_gaussian_data(linear_gaussian_ssm):
    """Synthetic trajectory + noisy observations from the reference SSM."""

    x0 = torch.tensor([1.5, 0.0])
    T = 80
    x_true, y = linear_gaussian_ssm.measure(x=x0, T=T, keep_y0=False, return_x=True)
    return {
        "x0": x0,
        "T": T,
        "x_true": x_true,
        "data": bside.Data(y=y),
        "init_dist": bside.FilteringDistribution(
            mean=x0,
            cov=0.01 * torch.eye(2),
        ),
    }
