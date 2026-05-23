"""
Multiple-shooting loss for system identification.

Wraps `SubspaceEncoder._loss` so it composes with any external training loop
or optimizer.  Typical usage::

    loss_fn = MultiShootingLoss(ssm=encoder, T=25)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for traj_batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(traj_batch)
        loss.backward()
        optimizer.step()
"""

from typing import Callable

import torch
from torch import Tensor

from bside.dataset import DataTrajectories
from bside.subspace_encoder import SubspaceEncoder


class MultiShootingLoss:
    """
    Composable multi-step prediction loss.

    Parameters
    ----------
    ssm : SubspaceEncoder
        The model to fit. Must expose ``_loss(T, trajectories, loss_fctn)``.
    T : int
        Number of simulation steps per shooting window.
    loss_fn : callable, optional
        Per-timestep loss applied to ``(prediction, target)``. Defaults to MSE.
    """

    def __init__(
        self,
        ssm: SubspaceEncoder,
        T: int,
        loss_fn: Callable = torch.nn.MSELoss(),
    ) -> None:

        self.ssm = ssm
        self.T = T
        self.loss_fn = loss_fn

    def __call__(
        self,
        trajectories: DataTrajectories,
    ) -> Tensor:

        return self.ssm._loss(self.T, trajectories, self.loss_fn)
