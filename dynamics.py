import torch
from torch import Tensor
import filtering
from models import Matrix
from dataset import Data, DataTrajectories
from abc import ABC, abstractmethod
from typing import Union, Tuple, Type, Callable

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
        dist : Type[filtering.FilteringDistribution],
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
    

class AdditiveModel(Model):

    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        noise_cov : Matrix
    ):
        
        super().__init__(in_dim, out_dim)
        self._noise_cov = noise_cov

    @property
    def noise_cov(
        self
    ):

        return self._noise_cov.val

    @noise_cov.setter
    def noise_cov(
        self,
        value
    ):
        
        raise ValueError('The noise covariance matrix cannot be modified')


    @property
    def sqrt_noise_cov(
        self
    ):

        return self._noise_cov.sqrt

    @sqrt_noise_cov.setter
    def sqrt_noise_cov(
        self,
        value
    ):
        self._noise_cov.sqrt = value

    def _update_sqrt(
        self
    ):
        
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
        dist : Type[filtering.FilteringDistribution],
        u : Tensor = None,
        crossCov : bool = False
    ) -> Union[Type[filtering.FilteringDistribution], Tuple[filtering.FilteringDistribution, Tensor]]:
        
        """
        Filters a Gaussian state vector using the Kalman filter.
        
        This function applies the Kalman filter to estimate the mean and covariance of the state vector given its current mean and covariance, and optionally an input vector.

        Parameters
        ----------
        dist : Type[FilteringDistribution]
            The current distribution of the state vector.

        u : Tensor, optional
            Input vector (default is None). If provided, the filter considers the influence of the input on the state vector.

        crossCov : bool, optional
            Boolean flag indicating whether the cross-covariance should be computed (default is False). If True, the function returns the covariance between the input and the output of the filter.

        Returns
        -------
        Union[Type[FilteringDistribution], Tuple[Type[FilteringDistribution], Tensor]]
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
        f : Type[torch.nn.Module],
        noise_cov : Matrix,
        in_dim : int,
        out_dim : int
    ):
        
        super().__init__(in_dim, out_dim, noise_cov)
        self.f = f

    def forward(
        self,
        x: Tensor,
        u: Tensor = None
    ) -> Tensor:
        
        return self.f(x, u)

    def filter(
        self,
        dist : Type[filtering.FilteringDistribution],
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
        dynamics : Type[Model],
        observations : Type[Model] = None,
        encoder : Type[Model] = None,
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

    # def _encoder(
    #     self,
    #     x: Tensor,
    #     u: Tensor = None
    # ):
        
    #     return self.dynamics(x,u)
    
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
    
    def forward(
        self,
        data : Union[Data, DataTrajectories],
        T : int = None
    ) -> Tensor:
        
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
        x = self.encoder(data.y[y_idx].reshape(data.num_traj,-1), data.u[u_idx].reshape(data.num_traj,-1) if data.u is not None else None)
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
        Find fit function on line 244 here: \
            https://github.com/GerbenBeintema/deepSI/blob/master/deepSI/fit_systems/fit_system.py

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
        init_dist : Type[filtering.FilteringDistribution],
        dynamics : Type[Model],
        observations : Type[Model],
        u : Tensor = None
    ):
        
        super().__init__(dynamics, observations)
        self.dist = init_dist
        self.u = u
    
    def forward(
        self,
        data : Data,
        T : int = None
    ) -> Tensor:
        
        pass


    def log_marg_likelihood(
        self,
        data : Data,
        y0 : bool = False
    ) -> Tensor:
        
        T = data.size

        dist = self.dist

        if not y0:
            dist = self.observations.filter(dist, data.u[0])

        for t in range(T):
            dist, U = self.observations.filter(dist, data.u[t], crossCov=True)

            log_prob = 0.

            if t < T-1:
                dist = filtering.kalman_update(dist, U, diff, sqrtS)
                dist = self.dynamics.filter(dist, data.u[t+1])
                if dist is None:
                    return torch.tensor([-torch.inf])
                
        return log_prob