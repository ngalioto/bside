import torch
from torch import Tensor
import filtering
from bside.models import Matrix
from bside.dataset import Data, DataTrajectories
from abc import ABC, abstractmethod
from typing import Union, Tuple, Callable

class Model(ABC, torch.nn.Module):

    def __init__(
        self,
        in_dim : int,
        out_dim : int
    ):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        u: Tensor = None
    ) -> Tensor:
        
        pass

    @abstractmethod
    def filter(
        self,
        dist : filtering.FilteringDistribution,
        u : Tensor = None,
        crossCov : bool = False
    ):
        
        pass

    @abstractmethod
    def sample(
        self,
        x: Tensor,
        u: Tensor = None
    ) -> Tensor:
        
        pass

    def predict(
        self,
        x : Tensor,
        u : Tensor = None,
        T : int = 1,
        keep_x0 : bool = False
    ) -> Tensor:
        
        """
        Runs the model forward `T` timesteps.

        Parameters
        ---------
        x : Tensor
            state vector. Should be shape (B, d) or (d) where B is the batch size and d is the state dimension.
        u : Tensor, optional
            input vector. Should be shape (T, m), or (m) if T is 1.

        Returns
        ---------
        Tensor
            The output of the `T` compositions of the forward model.
        """

        batch = x.ndim > 1

        if u.ndim == 1:
            u = u.unsqueeze(0)
        elif u.shape[0] < T:
            raise ValueError(f'{T} timesteps specified, but only {u.shape[0]} inputs provided')

        x_out = torch.zeros(x.shape[0] if batch else 1, T, self.ydim)
        x_out[:, 0] = x.clone()

        for ii in range(T):
            x_out[:, ii+1] = self(x_out[:, ii], u[ii] if u is not None else None)

        if not keep_x0:
            x_out = x_out[:, 1:]

        return x_out if batch else x_out.squeeze(0)
    

class AdditiveModel(Model):

    """
    TODO: Add function to run the filter over multiple timesteps.
    """

    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        noise_cov : Matrix
    ):
        
        super().__init__(in_dim, out_dim)

        if type(noise_cov) is Tensor:
            noise_cov = Matrix(noise_cov)
        elif type(noise_cov) is not Matrix:
            raise ValueError(f'`noise_cov` must be a Tensor or Matrix, but received {type(noise_cov)}')
        
        self._noise_cov = noise_cov

    @property
    def noise_cov(
        self
    ):

        return self._noise_cov.val

    @noise_cov.setter
    def noise_cov(
        self,
        value : Tensor
    ):
        
        self._noise_cov.val = value

    def update(
        self
    ):
        
        self._noise_cov.update()

    """Not sure we need the next three methods"""
    @property
    def sqrt_noise_cov(
        self
    ):

        return self._noise_cov.sqrt

    @sqrt_noise_cov.setter
    def sqrt_noise_cov(
        self,
        value : Tensor
    ):
        self._noise_cov.sqrt = value

    def _update_sqrt(
        self
    ):
        """
        Need some way to handle when Parameters are updated without using the setter
        """
        
        self.sqrt_noise_cov = torch.linalg.cholesky(self.noise_cov, upper=False)

    def sample(
        self,
        N : int = 1,
        x : Tensor = None,
        u : Tensor = None
    ) -> Tensor:
        
        x = self.x if x is None else x
        return torch.randn(N, self.xdim) @ self.sqrt_noise_cov.T + self.forward(x,u)

class LinearModel(AdditiveModel):
    """
    A model that is a linear function of the state and/or inputs. \
        The noise is assumed to be additive Gaussian.

    Attributes
    ---------
    xdim : int
        Input dimension of the model
    ydim : int
        Output dimension of the model
    mat_x : Tensor
        A matrix that will be applied to the state within the model
    mat_u : Tensor
        A matrix that will be applied to the input within the model. If the model does not use inputs, this attribute will be set as `None`
    noise_cov : Tensor
        The covariance matrix of the additive noise term in the model. This matrix must be symmetric positive-definite
    sqrt_noise_cov : Tensor
        The lower Cholesky decomposition of `noise_cov`. This will be useful for sampling and evaluating log pdfs.
    """

    def __init__(
        self,
        mat_x : Matrix,
        noise_cov : Matrix,
        mat_u : Matrix = None
    ):
        """
        Constructor method for LinearModel class

        Parameters
        ---------
        mat_x : Tensor
            matrix that transforms the state vector
        noise_cov : Tensor
            noise covariance matrix. Must be symmetric positive-definite.
        mat_u : Tensor, optional
            matrix that transforms the input vector
        """

        if mat_x.val.ndim != 2:
            raise ValueError(f'`mat_x` must have two dimensions, but has {mat_x.val.ndim}')

        if mat_u is not None:
            if mat_u.val.ndim != 2:
                raise ValueError(f'`mat_u` must have two dimensions, but has {mat_u.val.ndim}')
            
            if mat_u.val.shape[0] != mat_x.val.shape[0]:
                raise ValueError('The dimensions of mat_x and mat_u at axis 1 must agree')

        super().__init__(*mat_x.val.shape, noise_cov)

        self._mat_x = mat_x
        self._mat_u = mat_u
        self.indices = torch.unique(torch.cat((
            mat_x.indices,
            noise_cov.indices
        ))) # assumes mat_u is known if not None

    def update(
        self
    ):
        
        super().update()
        self._mat_x.update()
        self._mat_u.update()


    @property
    def mat_x(
        self
    ):

        return self._mat_x.val

    @mat_x.setter
    def mat_x(
        self,
        value
    ):
        raise ValueError('The matrix `mat_x` cannot be modified')

    @property
    def mat_u(
        self
    ):

        return self._mat_u.val

    @mat_u.setter
    def mat_u(
        self,
        value
    ):
        raise ValueError('The matrix `mat_u` cannot be modified')

    def forward(
        self,
        x : Tensor,
        u : Tensor = None
    ) -> Tensor:
        
        """
        Evaluates the deterministic component of the function

        Parameters
        ---------
        x : Tensor
            state vector. 
        u : Tensor, optional
            input vector. 

        Returns
        ---------
        Tensor
            The output of the forward model without noise.
        """

        x_next = torch.einsum('ij,bj->bi', self.mat_x, x)

        if self._mat_u is not None and u is not None:
            x_next = x_next + self.mat_u @ u

        return x_next


    def filter(
        self,
        dist : filtering.FilteringDistribution,
        u : Tensor = None,
        crossCov : bool = False
    ) -> Union[filtering.FilteringDistribution, Tuple[filtering.FilteringDistribution, Tensor]]:
        
        """
        Filters a Gaussian state vector using the Kalman filter.
        
        This function applies the Kalman filter to estimate the mean and covariance of the state vector given its current mean and covariance, and optionally an input vector.

        Parameters
        ----------
        dist : FilteringDistribution
            The current distribution of the state vector.

        u : Tensor, optional
            Input vector (default is None). If provided, the filter considers the influence of the input on the state vector.

        crossCov : bool, optional
            Boolean flag indicating whether the cross-covariance should be computed (default is False). If True, the function returns the covariance between the input and the output of the filter.

        Returns
        -------
        Union[FilteringDistribution, Tuple[FilteringDistribution, Tensor]]
            The filtered distribution of the state vector. If `crossCov` is True, the function returns a tuple of the filtered distribution and the cross-covariance.
        """

        return filtering.kf_predict(self, dist, u, crossCov)
    
class IdentityModel(LinearModel):

    def __init__(
        self,
        dim : int
    ):
        
        super().__init__(
            mat_x=Matrix(torch.eye(dim)),
            noise_cov=Matrix(torch.eye(dim))
        )

    def forward(
        self,
        x: Tensor,
        u: Tensor = None
    ) -> Tensor:
        
        return x
    
class NonlinearModel(AdditiveModel):

    def __init__(
        self,
        f : torch.nn.Module,
        noise_cov : Matrix,
        in_dim : int,
        out_dim : int
    ):
        
        super().__init__(in_dim, out_dim, noise_cov)
        self.f = f

    def update(
        self
    ):
        
        self.noise_cov.update()

    def forward(
        self,
        x: Tensor,
        u: Tensor = None
    ) -> Tensor:
        
        return self.f(x, u)

    def filter(
        self,
        dist : filtering.FilteringDistribution,
        u : Tensor = None,
        crossCov : bool = False
    ):
        
        return filtering.enkf_predict(
            self,
            dist=dist,
            u=u,
            crossCov=crossCov
        )
    
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
            The log marginal likelihood of the data given the model $\log p(\mathcal{Y}_n | \theta)$.
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