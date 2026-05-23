import torch
from torch import Tensor
from typing import Callable, Tuple, Union

from bside.ssm import SSM
from bside.dynamics import Model
from bside.models import ResidualNetwork
from bside.dataset import Data, DataTrajectories


class SubspaceEncoder(SSM):

    """
    Implements a deep subspace encoder from the paper:
      Gerben Beintema, Roland Toth, Maarten Schoukens.
      Nonlinear State-Space Identification using Deep Encoder Networks;
      Proceedings of the 3rd Conference on Learning for Dynamics and Control,
      PMLR 144:241-250, 2021.

      https://www.sciencedirect.com/science/article/pii/S0005109823003710

    Adds an encoder module on top of the bare `SSM` interface and provides a
    multiple-shooting training loop that minimizes simulation error over short
    rollouts inside each trajectory.
    """

    def __init__(
        self,
        xdim : int = 10,
        ydim : int = 10,
        udim : int = 1,
        num_y_hist : int = 20,
        num_u_hist : int = 20,
        feedthrough : bool = False,
        dynamics : Model | None = None,
        observations : Model | None = None,
        encoder : Model | None = None,
    ):
        
        """
        Parameters
        ----------
        xdim : int, optional
            Dimension of the latent state. Defaults to 10.
        ydim : int, optional
            Dimension of the output. Defaults to 10.
        udim : int, optional
            Dimension of the control input. Defaults to 1.
        num_y_hist : int, optional
            Number of past observations fed into the encoder.
        num_u_hist : int, optional
            Number of past inputs fed into the encoder.
        feedthrough : bool, optional
            Whether the system has a direct feedthrough from inputs to outputs.
        dynamics, observations, encoder : Model, optional
            Custom subnetworks. If omitted, default `ResidualNetwork`s are
            constructed with 2 hidden layers of width 64.
        """

        if encoder is None:
            encoder = ResidualNetwork(
                n_in=ydim * num_y_hist + udim * num_u_hist,
                n_out=xdim,
                n_hidden=64,
                n_layers=2,
            )
        if dynamics is None:
            dynamics = ResidualNetwork(
                n_in=xdim + udim,
                n_out=xdim,
                n_hidden=64,
                n_layers=2,
            )
        if observations is None:
            observations = ResidualNetwork(
                n_in=xdim + udim if feedthrough else xdim,
                n_out=ydim,
                n_hidden=64,
                n_layers=2,
            )

        super().__init__(
            xdim=xdim,
            ydim=ydim,
            dynamics=dynamics,
            observations=observations,
        )

        self.encoder = encoder
        self.num_y_hist = num_y_hist
        self.num_u_hist = num_u_hist
        self.history_length = max(num_y_hist, num_u_hist)

    def update(
        self,
        params : Tensor | None = None
    ) -> None:
        
        super().update(params)
        if hasattr(self.encoder, 'update') and callable(getattr(self.encoder, 'update')):
            try:
                self.encoder.update(params)
            except TypeError:
                # Encoder is a plain nn.Module without our update(params) hook.
                pass

    def forward(
        self,
        data : Union[Data, DataTrajectories],
        T : int = None,
    ) -> Tensor:
        
        """
        Run the multiple-shooting forward pass over a batch of trajectories.

        For each trajectory the first `history_length` samples are fed into the
        encoder to produce an initial latent state, and then the dynamics are
        rolled out for at most `T` steps.  Trajectories that finish before
        `T` steps are masked out so the resulting tensor of outputs is dense.
        """
        
        data = DataTrajectories(batch=[data]) if not isinstance(data, DataTrajectories) else data
        if T is None:
            T = data.max_length - self.history_length
        else:
            data = data.partition_trajectories(T, self.history_length)

        y = torch.zeros(data.num_traj, T, self.ydim)

        # Indices for the time histories fed to the encoder
        range_y = torch.arange(0, self.num_y_hist).expand(data.num_traj, self.num_y_hist)
        y_idx = (range_y + data.start_indices.unsqueeze(1)).flatten()
        range_u = torch.arange(0, self.num_u_hist).expand(data.num_traj, self.num_u_hist)
        u_idx = (range_u + data.start_indices.unsqueeze(1)).flatten()

        # Initial latent state from the encoder
        x = self.encoder(
            data.y[y_idx].reshape(data.num_traj, -1),
            data.u[u_idx].reshape(data.num_traj, -1) if data.u is not None else None,
        )
        y[:, 0] = self.observations(x)

        # Mask-based progressive update: only propagate trajectories that have not
        # yet reached their end.  Preserves the existing `remaining_traj` trick
        # so we don't waste compute on trajectories that finished early.
        for ii in range(1, T):
            remaining_traj = ii < (data.traj_lengths - self.history_length)
            x[remaining_traj] = self.dynamics(
                x[remaining_traj],
                data.u[data.start_indices[remaining_traj] + self.history_length + ii]
                if data.u is not None else None,
            )
            y[remaining_traj, ii] = self.observations(x[remaining_traj])

        # Trim trajectories that ended before T steps
        range_tensor = torch.arange(0, T).expand(data.num_traj, T)
        mask = (range_tensor < (data.traj_lengths - self.history_length).unsqueeze(1)).reshape(-1)

        return y.reshape(-1, self.ydim)[mask]

    def _loss(
        self,
        T : int,
        data : DataTrajectories,
        loss_fctn : Callable,
    ) -> Tensor:
        
        outputs = self(data, T)

        range_tensor = torch.arange(0, data.max_length).expand(data.num_traj, data.max_length)
        mask = torch.logical_and(
            range_tensor >= self.history_length,
            range_tensor < data.traj_lengths.unsqueeze(1),
        )
        target_idx = (range_tensor + data.start_indices.unsqueeze(1))[mask]

        return loss_fctn(outputs, data.y[target_idx])

    def fit(
        self,
        training_data : DataTrajectories,
        validation_data : DataTrajectories = None,
        T : int = -1,
        loss_fctn : Callable = torch.nn.MSELoss(),
        epochs : int = 30,
        batch_size : int = 256,
        normalize : bool = True,
        shuffle : bool = True,
        ms_batching : bool = False,
        verbose : bool = True,
        **optim_kwargs,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        
        """
        Train the subspace encoder via multiple-shooting on simulation error.

        Parameters
        ----------
        training_data, validation_data : DataTrajectories
            Trajectories to fit on (and optionally validate against).
        T : int, optional
            Number of simulation steps per shooting window.  ``-1`` uses the
            longest trajectory in the dataset.
        loss_fctn : Callable, optional
            Loss function on the (prediction, target) pair.  Defaults to MSE.
        epochs : int, optional
        batch_size : int, optional
        normalize : bool, optional
            Z-score the training data before fitting (idempotent).
        shuffle : bool, optional
            Shuffle the trajectory order each epoch.
        ms_batching : bool, optional
            Pre-partition trajectories into shooting windows before training.
        verbose : bool, optional
            Print per-epoch training/validation loss.
        **optim_kwargs
            Passed to `torch.optim.Adam`.
        """

        training_data.normalize() if normalize else training_data.unnormalize()
        if validation_data is not None:
            validation_data.normalize() if normalize else validation_data.unnormalize()

        if T == -1:
            T = training_data.max_length if validation_data is None else max(training_data.max_length, validation_data.max_length)
            T -= self.history_length
        elif T < 1:
            raise ValueError(f"Time horizon T must be greater than 0, but received value {T}")

        min_length = training_data.min_length if validation_data is None else min(training_data.min_length, validation_data.min_length)
        if self.history_length + 1 > min_length:
            raise ValueError(
                f"The minimum trajectory length is {min_length}, but the encoder requires "
                f"a time history of at least {self.history_length + 1} data points. "
                f"Current parameter values are num_u_hist={self.num_u_hist} and num_y_hist={self.num_y_hist}."
            )

        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)

        if ms_batching:
            training_data = training_data.partition_trajectories(T, self.history_length)
            if validation_data is not None:
                validation_data = validation_data.partition_trajectories(T, self.history_length)

        total_loss = torch.zeros(epochs)
        training_loader = torch.utils.data.DataLoader(
            dataset=training_data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: DataTrajectories(batch=x),
        )

        if validation_data is not None:
            total_vloss = torch.zeros(epochs)
            validation_loader = torch.utils.data.DataLoader(
                dataset=validation_data,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: DataTrajectories(batch=x),
            )

        for ii in range(epochs):
            if verbose:
                print(f"Epoch {ii + 1}:")
            self.train()
            for trajectories in training_loader:
                loss = self._loss(T, trajectories, loss_fctn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.update()

                total_loss[ii] += loss.item() / len(training_loader)

            if verbose:
                print(f"  Training Loss: {total_loss[ii]}")

            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    for vtrajectories in validation_loader:
                        vloss = self._loss(T, vtrajectories, loss_fctn)
                        total_vloss[ii] += vloss / len(validation_loader)
                if verbose:
                    print(f"  Validation Loss: {total_vloss[ii]}")

        return (total_loss, total_vloss) if validation_data is not None else total_loss
