import torch
from torch import Tensor
import bside.filtering as filtering
from bside.dynamics import Model, IdentityModel
from bside.dataset import Data, DataTrajectories
from typing import Union, Tuple, Callable


class SSM(torch.nn.Module):

    """
    A state-space model (SSM) that is defined by a dynamics model and an observation model. \
        The dynamics model is a function that maps the state vector to the next state vector, and the observation model is a function that maps the state vector to the observation vector.
    """

    def __init__(
        self,
        xdim : int,
        ydim : int,
        dynamics : Model,
        observations : Model = None,
        encoder : Model = None,
        num_y_hist : int = 1,
        num_u_hist : int = 1
    ):
        
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim

        if observations is None and xdim != ydim:
            raise ValueError(f"Observation model must be provided if xdim != ydim. Received xdim = {xdim} and ydim = {ydim}")

        self.encoder = encoder if encoder is not None else dynamics
        self.dynamics = dynamics
        self.observations = IdentityModel(self.xdim) if observations is None else observations

        self.num_y_hist = num_y_hist
        self.num_u_hist = num_u_hist
        self.history_length = max(num_y_hist, num_u_hist)
    
    def __repr__(
        self
    ) -> str:
        
        name = "State-space model (SSM):\n"
        xdim = f"  State dimension: {self.xdim}\n"
        ydim = f"  Output dimension: {self.ydim}\n"
        dynamics = f"  Dynamics:\n    {self.dynamics}\n"
        observations = f"  Observations:\n    {self.observations}\n"
        encoder = f"  Encoder:\n    {self.encoder if self.encoder is not self.dynamics else None}\n"
        return name + xdim + ydim + dynamics + observations + encoder
    
    def predict(
        self,
        x : Tensor,
        u : Tensor = None,
        T : int = 1,
        return_x : bool = False,
        keep_y0 : bool = False
    ):
        
        x = self.dynamics.predict(x, u, T, keep_x0=keep_y0)
        y = self.observations(x)

        return y if not return_x else (x, y)

    
    def forward(
        self,
        data : Union[Data, DataTrajectories],
        T : int = None
    ) -> Tensor:
        
        """
        Compute the output of the SSM given the input data.

        If you want to use this to predict from a single initial condition, run `forward()`

        Parameters
        ----------
        data : Union[Data, DataTrajectories]
            The data to be used for training.
        T : int, optional
            The time horizon for the multiple shooting implementation. If not provided, the time horizon is set to the max trajectory length.

        Returns
        -------
        Tensor
            The output of the SSM given the input data.
        """
        
        data = DataTrajectories(batch=[data]) if not isinstance(data, DataTrajectories) else data
        if T is None:
            T = data.max_length - self.history_length # max trajectory length
        else:
            data = data.partition_trajectories(T, self.history_length)

        # initialize the output tensor
        y = torch.zeros(data.num_traj, T, self.ydim)

        range_tensor = torch.arange(0, self.num_y_hist).expand(data.num_traj, self.num_y_hist)
        y_idx = (range_tensor + data.start_indices.unsqueeze(1)).flatten()
        range_tensor = torch.arange(0, self.num_u_hist).expand(data.num_traj, self.num_u_hist)
        u_idx = (range_tensor + data.start_indices.unsqueeze(1)).flatten()

        # indices to get a batch of time histories for y and u
        # y_idx = torch.stack([torch.arange(data.traj_indices[ii], data.traj_indices[ii]+self.num_y_hist) for ii in range(data.num_traj)])
        # u_idx = torch.stack([torch.arange(data.traj_indices[ii], data.traj_indices[ii]+self.num_u_hist) for ii in range(data.num_traj)])

        # compute intial conditions using time histories of data
        # QUESTION: IS THIS HOW WE WANT TO PASS IN THE Y AND U?
        x = self.encoder(
            data.y[y_idx].reshape(data.num_traj,-1), 
            data.u[u_idx].reshape(data.num_traj,-1) if data.u is not None else None
        )
        y[:,0] = self.observations(x)
        
        # indices where a trajectory has not been fully computed
        remaining_traj = torch.arange(data.num_traj)
        for ii in range(1, T):
            remaining_traj = ii < (data.traj_lengths-self.history_length)
            x[remaining_traj] = self.dynamics(x[remaining_traj], 
                                data.u[data.start_indices[remaining_traj] + self.history_length + ii] 
                                if data.u is not None else None)
            y[remaining_traj, ii] = self.observations(x[remaining_traj])
                
        # indices excluding points where no computed trajectory exists
        range_tensor = torch.arange(0, T).expand(data.num_traj, T)
        mask = (range_tensor < (data.traj_lengths - self.history_length).unsqueeze(1)).reshape(-1)

        # trim_idx = torch.stack([torch.arange(start, start+data.traj_lengths[ii]-self.history_length) for ii, start in enumerate(range(0, data.num_traj*T, T))])

        # reshape and trim the output tensor to match the target data shape
        return y.reshape(-1, self.ydim)[mask]
    
    def _loss(
        self,
        T : int,
        data : DataTrajectories,
        loss_fctn : Callable
    ) -> Tensor:
        
        """
        Compute the loss for the multiple shooting implementation.

        Parameters
        ----------
        T : int
            The time horizon for the multiple shooting implementation.
        data : DataTrajectories
            The data to be used for training.
        loss_fctn : Callable
            The loss function to be used for training.

        Returns 
        -------
        Tensor
            The loss for the multiple shooting implementation.
        """

        outputs = self(data, T)

        # target_idx = torch.cat([torch.arange(start, end) for start, end in zip(data.start_indices + self.history_length, data.traj_indices[1:])])
        range_tensor = torch.arange(0, data.max_length).expand(data.num_traj, data.max_length)
        mask = torch.logical_and(range_tensor >= self.history_length, range_tensor < data.traj_lengths.unsqueeze(1))
        target_idx = (range_tensor + data.start_indices.unsqueeze(1))[mask]

        # return torch.mean((outputs - data.y[target_idx])**2)
        return loss_fctn(outputs, data.y[target_idx])
    
    def update(
        self
    ):
        
        """
        Map the updated parameters into structured matrices (Matrix).
        """

        self.encoder.update()
        self.dynamics.update()
        self.observations.update()
    
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
        **optim_kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        
        """
        Parameters
        ----------
        epochs : int, optional
            The number of epochs to train the subspace encoder, by default 30
        batch_size : int, optional
            The batch size to use during training, by default 256
        loss_kwargs : dict, optional
            The keyword arguments to pass to the loss function, by default {'nf':25, 'stride':1}
                nf : int, optional
                    The number of future steps to predict, by default 25
                stride : int, optional
                    The number of steps to skip between training points, by default 1
        **kwargs
            Additional keyword arguments to pass to the deep subspace encoder training function.

        Raises
        ------
        ValueError
            If the number of training data points is less than the number of data points required by the encoder.
        """

        # normalize the data if necessary
        training_data.normalize() if normalize else training_data.unnormalize()
        if validation_data is not None:
            validation_data.normalize() if normalize else validation_data.unnormalize

        # set the time horizon T to the max value if T = -1. Otherwise, check that T is valid
        if T == -1:
            T = training_data.max_length if validation_data is None else max(T, validation_data.max_length)
            T -= self.history_length
        elif T < 1:
            raise ValueError(f"Time horizon T must be greater than 0, but received value {T}")

        # check if the data is long enough for the encoder
        min_length = training_data.min_length if validation_data is None else min(training_data.min_length, validation_data.min_length)        
        if self.history_length + 1 > min_length:
            raise ValueError(f"The minimum trajectory length is {min_length}, but the encoder requires a time history of at least {self.history_length + 1} data points. " +
                             f"The hyperparameters must satisfy max(na, nb) + 1 <= {min_length}. " +
                             f"Current parameter values are na = {self.num_u_hist} and nb = {self.num_y_hist}.")
        
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
            collate_fn=lambda x : DataTrajectories(batch=x)
        )

        if validation_data is not None:
            total_vloss = torch.zeros(epochs)
            validation_loader = torch.utils.data.DataLoader(
                dataset=validation_data, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=lambda x : DataTrajectories(batch=x)
            )

        for ii in range(epochs):
            print(f"Epoch {ii + 1}:")
            self.train()
            for trajectories in training_loader:
                loss = self._loss(T, trajectories, loss_fctn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.update()

                total_loss[ii] += loss.item() / len(training_loader)

            print(f"  Training Loss: {total_loss[ii]}")

            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    for vtrajectories in validation_loader:
                        vloss = self._loss(T, vtrajectories, loss_fctn)
                        total_vloss[ii] += vloss / len(validation_loader)
                print(f"  Validation Loss: {total_vloss[ii]}")
        
        return (total_loss, total_vloss) if validation_data is not None else total_loss
    
    
class HMM(SSM):

    """
    A hidden Markov model (HMM) that is defined by a dynamics model and an observation model. \
        The dynamics model is a function that maps the state vector to the next state vector, and the observation model is a function that maps the state vector to the observation vector.
    """

    def __init__(
        self,
        init_dist : filtering.FilteringDistribution,
        dynamics : Model,
        observations : Model,
        u : Tensor = None
    ):
        
        super().__init__(dynamics, observations)
        self.init_dist = init_dist
        self.u = u

    def update(
        self
    ):
        
        self.init_dist.update()
        self.dynamics.update()
        self.observations.update()
    
    def forward(
        self,
        data : Data,
        T : int = None
    ) -> Tensor:
        
        pass

    def _loss(
        self,
        data : Data,
        y0 : bool = False
    ) -> Tensor:
        
        """
        This function should be able to compute log_marg_likelihood over multiple trajectories.

        Then can use inherited `fit` method to train.
        """
        
        return self.log_marg_likelihood(data, y0)

    def log_marg_likelihood(
        self,
        data : Data,
        y0 : bool = False
    ) -> Tensor:
        
        """
        NOTE: This does not run properly yet.

        TODO: How to handle the data.u depending on y0. Might have to change data structure.
        If y0 is False, then we need one more u than y.
        Maybe add y0 flag to Data and pad the first y with zeros. Then alter the __getitem__ to return None if 0 in slice.

        Parameters
        ----------
        data : Data
            The data to be used for training. For now, must be a single trajectory.
        y0 : bool, optional
            Boolean flag indicating whether the data is available on the initial condition.
        
        Returns
        -------
        Tensor
            The log marginal likelihood of the data given the model $\\log p(\mathcal{Y}_n | \\theta)$.
        """
        
        T = data.size

        x_dist = self.init_dist.copy()

        if not y0:
            x_dist = self.dynamics.filter(
                x_dist, 
                data.u[0] if data.u is not None else None
            )
            if x_dist is None:
                return torch.tensor([-torch.inf])

        for t in range(0 if y0 else 1, T):
            y_dist, U = self.observations.filter(
                x_dist, 
                data.u[t] if data.u is not None else None, 
                crossCov=True
            )

            log_prob = y_dist.log_prob(data.y[t])

            if t < T-1:
                # need an update that will also work for particle methods
                x_dist = filtering.kalman_update(data.y[t], x_dist, y_dist, U)
                x_dist = self.dynamics.filter(
                    x_dist, 
                    data.u[t+1] if data.u is not None else None
                )
                if x_dist is None:
                    return torch.tensor([-torch.inf])
                
        return log_prob