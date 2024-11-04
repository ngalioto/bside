import torch
from torch import Tensor
from bside.models import Matrix
from abc import ABC, abstractmethod

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