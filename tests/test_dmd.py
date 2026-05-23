"""DMD / DMDc tests."""

import pytest
import torch

import bside


def test_dmd_recovers_known_operator():
    A_true = torch.tensor([[0.95, 0.1], [-0.1, 0.9]])
    x = torch.zeros(2, 200)
    x[:, 0] = torch.tensor([1.0, 0.5])
    for t in range(199):
        x[:, t + 1] = A_true @ x[:, t]

    dmd = bside.DMD()
    dmd.estimate_model(input=x[:, :-1], output=x[:, 1:], rank=2, rom=False)
    assert torch.allclose(dmd.A, A_true, atol=1e-4)


def test_dmd_rom_branch():
    A_true = torch.tensor([[0.95, 0.1], [-0.1, 0.9]])
    x = torch.zeros(2, 200)
    x[:, 0] = torch.tensor([1.0, 0.5])
    for t in range(199):
        x[:, t + 1] = A_true @ x[:, t]

    dmd = bside.DMD()
    dmd.estimate_model(input=x[:, :-1], output=x[:, 1:], rank=2, rom=True)
    # Eigenvalues of the rom A should match the eigenvalues of A_true
    lam_rom = torch.sort(torch.linalg.eigvals(dmd.A_rom).abs())[0]
    lam_true = torch.sort(torch.linalg.eigvals(A_true).abs())[0]
    assert torch.allclose(lam_rom, lam_true, atol=1e-3)


def test_dmdc_recovers_A_and_B():
    A_true = torch.tensor([[0.9, 0.05], [0.0, 0.85]])
    B_true = torch.tensor([[0.1], [0.2]])
    T = 300
    x = torch.zeros(2, T)
    x[:, 0] = torch.tensor([0.5, -0.5])
    u = 0.5 * torch.sin(torch.linspace(0.0, 8 * torch.pi, T - 1)).unsqueeze(0)
    for t in range(T - 1):
        x[:, t + 1] = A_true @ x[:, t] + B_true @ u[:, t]

    dmdc = bside.DMDc()
    inp = torch.cat([x[:, :-1], u], dim=0)
    dmdc.estimate_model(input=inp, output=x[:, 1:], rank=3, rom=False)
    assert torch.allclose(dmdc.A, A_true, atol=1e-3)
    assert torch.allclose(dmdc.B, B_true, atol=1e-3)


def test_dmdc_rom_branch_runs():
    A_true = 0.9 * torch.eye(3)
    B_true = torch.tensor([[1.0], [0.0], [0.5]])
    T = 200
    x = torch.zeros(3, T)
    x[:, 0] = torch.tensor([1.0, 0.0, 0.5])
    u = 0.3 * torch.cos(torch.linspace(0.0, 6 * torch.pi, T - 1)).unsqueeze(0)
    for t in range(T - 1):
        x[:, t + 1] = A_true @ x[:, t] + B_true @ u[:, t]

    dmdc = bside.DMDc()
    inp = torch.cat([x[:, :-1], u], dim=0)
    dmdc.estimate_model(input=inp, output=x[:, 1:], rank=3, rom=True)
    assert dmdc.A_rom.shape == (3, 3)
    assert dmdc.B_rom.shape == (3, 1)
