import torch
from torch import Tensor
from bside.models import Matrix, PSDMatrix
from abc import ABC, abstractmethod
from typing import Callable
import warnings


"""
The building blocks for creating a (possibly stochastic) dynamical system.

Three concepts cooperate here:

* `Model` is the abstract interface every block satisfies: it has a deterministic
  ``forward(x, u)`` and an ``update(params)`` hook that propagates learnable
  parameters into nested `Matrix` / `PSDMatrix` containers.
* `LinearModel` and `NonlinearModel` provide the deterministic dynamics
  themselves (matrices for the former, an arbitrary callable / `nn.Module`
  for the latter).
* `AdditiveModel` wraps any `Model` with additive zero-mean noise of a chosen
  PSD covariance. `LinearGaussianModel` and `NonlinearAdditiveModel` are
  convenience subclasses that pre-package the deterministic + noise pair.

The wrapping is implemented via plain `nn.Module` composition (the inner
deterministic model lives on ``self.model``) so PyTorch's parameter and
submodule registration works without surprises.
"""


class Model(torch.nn.Module, ABC):

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

    def update(
        self,
        params: Tensor | None = None
    ) -> None:
        """Default no-op. Subclasses with learnable matrices override this."""
        return None

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
        T : int, optional
            number of timesteps to run the model forward.
        keep_x0 : bool, optional
            if True, the initial state `x` will be included in the output. Otherwise, the output will only include the predicted states.

        Returns
        ---------
        Tensor
            The output of the `T` compositions of the forward model. Will have `T` timesteps if `keep_x0` is False, and `T+1` timesteps if `keep_x0` is True.
        """

        batch = x.ndim > 1

        if u is not None:
            if u.ndim == 1:
                u = u.unsqueeze(0)
            elif u.shape[0] < T:
                raise ValueError(f'{T} timesteps specified, but only {u.shape[0]} inputs provided')

        x_out = torch.zeros(x.shape[0] if batch else 1, T+1, self.out_dim)
        x_out[:, 0] = x.clone() if batch else x.unsqueeze(0)

        for ii in range(T):
            x_out[:, ii+1] = self(x_out[:, ii], u[ii] if u is not None else None)

        if not keep_x0:
            x_out = x_out[:, 1:]

        return x_out if batch else x_out.squeeze(0)
    

    def sample(
        self,
        x : Tensor,
        u : Tensor = None,
        N : int = 1
    ) -> Tensor:
        
        warnings.warn('The model is deterministic, so calling `sample` is equivalent to calling `forward`', UserWarning)
        
        return self.forward(x, u)

class LinearModel(Model):
    """
    A linear model.
    """

    def __init__(
        self,
        mat_x : Matrix,
        mat_u : Matrix = None
    ):
        """
        Constructor method for LinearModel class

        Parameters
        ---------
        mat_x : Matrix
            matrix that transforms the state vector
        mat_u : Matrix, optional
            matrix that transforms the input vector
        """

        if mat_x.val.ndim != 2:
            raise ValueError(f'`mat_x` must have two dimensions, but has {mat_x.val.ndim}')

        if mat_u is not None:
            if mat_u.val.ndim != 2:
                raise ValueError(f'`mat_u` must have two dimensions, but has {mat_u.val.ndim}')
            
            if mat_u.val.shape[0] != mat_x.val.shape[0]:
                raise ValueError('The dimensions of mat_x and mat_u at axis 1 must agree')

        super().__init__(
            in_dim=mat_x.val.shape[1],
            out_dim=mat_x.val.shape[0]
        )

        self._mat_x = mat_x
        self._mat_u = mat_u
        self.indices = torch.unique(mat_x.indices)

    def update(
        self,
        params: Tensor | None = None
    ) -> None:
        
        self._mat_x.update(params)
        if self._mat_u is not None:
            self._mat_u.update(params)

    @property
    def mat_x(
        self
    ) -> Tensor:

        return self._mat_x.val

    @mat_x.setter
    def mat_x(
        self,
        value
    ) -> None:
        raise ValueError('The matrix `mat_x` cannot be modified')

    @property
    def mat_u(
        self
    ) -> Tensor | None:

        return None if self._mat_u is None else self._mat_u.val

    @mat_u.setter
    def mat_u(
        self,
        value
    ) -> None:
        raise ValueError('The matrix `mat_u` cannot be modified')

    def forward(
        self,
        x : Tensor,
        u : Tensor = None
    ) -> Tensor:
        
        """
        Evaluates the deterministic component of the function via the batched mat-vec einsum
        pattern '...ij,...j->...i', so a single LinearModel handles non-batched,
        batched, sigma-point, and ensemble inputs uniformly.
        """

        x_next = torch.einsum('...ij,...j->...i', self.mat_x, x)

        if self._mat_u is not None and u is not None:
            x_next = x_next + torch.einsum('...ij,...j->...i', self.mat_u, u)

        return x_next


class NonlinearModel(Model):
    """A nonlinear deterministic model defined by an arbitrary callable or `nn.Module`.

    `nn.Module.__setattr__` already registers a `Module`-valued attribute as a
    submodule (so its parameters propagate to ``self.parameters()``) and stores
    plain callables in `__dict__`, so we don't need any special bookkeeping here.
    """

    def __init__(
        self,
        f : Callable[[Tensor, Tensor], Tensor] | torch.nn.Module,
        in_dim : int,
        out_dim : int
    ):
        
        super().__init__(in_dim, out_dim)
        self.f = f

    def update(
        self,
        params: Tensor | None = None
    ) -> None:
        # Delegate to the inner callable if it exposes an update hook.
        if hasattr(self.f, 'update') and callable(self.f.update):
            self.f.update(params)

    def forward(
        self,
        x: Tensor,
        u: Tensor = None
    ) -> Tensor:

        return self.f(x, u)


class IdentityModel(LinearModel):
    """Identity dynamics / observations: returns its input."""

    def __init__(
        self,
        dim : int
    ):
        
        super().__init__(
            mat_x=Matrix(torch.eye(dim))
        )

    def forward(
        self,
        x: Tensor,
        u: Tensor = None
    ) -> Tensor:
        
        return x


class AdditiveModel(Model):
    """
    Wraps a deterministic `Model` with additive zero-mean noise of a chosen PSD
    covariance.  Only the first two moments of the noise are used because that is
    all Gaussian filters need; subclasses (e.g. `NonlinearAdditiveModel`) can
    override `sample` if non-Gaussian noise is required.

    The inner deterministic model lives on ``self.model``; properties and methods
    forward to it where appropriate so that downstream filter code can treat
    ``AdditiveModel`` (or its subclasses) as if it were the inner model plus a
    noise covariance.
    """

    def __init__(
        self,
        model : Model,
        noise_cov : Tensor | PSDMatrix
    ):
        
        if not isinstance(model, Model):
            raise ValueError(f'`model` must be a `Model` instance, got {type(model)}.')

        super().__init__(in_dim=model.in_dim, out_dim=model.out_dim)

        # Proper nn.Module composition (registers `model` as a submodule so its
        # learnable parameters are discoverable via self.parameters()).
        self.model = model

        if isinstance(noise_cov, Tensor):
            noise_cov = PSDMatrix(noise_cov)
        elif not isinstance(noise_cov, PSDMatrix):
            raise ValueError(f'`noise_cov` must be a Tensor or PSDMatrix, got {type(noise_cov)}')
        self._noise_cov = noise_cov

    @property
    def noise_cov(
        self
    ) -> Tensor:

        return self._noise_cov.val

    @noise_cov.setter
    def noise_cov(
        self,
        value : Tensor
    ) -> None:
        
        self._noise_cov.val = value

    @property
    def sqrt_noise_cov(
        self
    ) -> Tensor:

        return self._noise_cov.sqrt

    @sqrt_noise_cov.setter
    def sqrt_noise_cov(
        self,
        value : Tensor
    ) -> None:
        self._noise_cov.sqrt = value

    def update(
        self,
        params: Tensor | None = None
    ) -> None:
        
        self.model.update(params)
        self._noise_cov.update(params)

    def forward(
        self,
        x : Tensor,
        u : Tensor = None
    ) -> Tensor:
        
        return self.model(x, u)

    def sample(
        self,
        x : Tensor,
        u : Tensor = None,
        N : int | None = None
    ) -> Tensor:
        
        """
        Draw a sample from the additive noise model: ``f(x, u) + eta`` with
        ``eta ~ N(0, Q)``.

        If ``N`` is None, draws one independent noise sample per row of ``x``
        (the usual case for ensemble / particle propagation). If ``N`` is
        provided, draws ``N`` total samples (broadcasting the deterministic part
        across them).
        """
        
        if N is None:
            N = x.shape[0] if x.ndim > 1 else 1
        noise = torch.randn(N, self.out_dim) @ self.sqrt_noise_cov.T
        return noise + self.forward(x, u)


class LinearGaussianModel(AdditiveModel):
    """Linear deterministic dynamics with additive Gaussian noise."""

    def __init__(
        self,
        model : LinearModel | None = None,
        noise_cov : Tensor | PSDMatrix | None = None,
        mat_x : Matrix | None = None,
        mat_u : Matrix | None = None,
    ):

        if model is None:
            if mat_x is None:
                raise ValueError('Provide either `model` or `mat_x` to construct a LinearGaussianModel.')
            model = LinearModel(mat_x=mat_x, mat_u=mat_u)
        elif not isinstance(model, LinearModel):
            raise ValueError(f'`model` must be a LinearModel for LinearGaussianModel, got {type(model)}.')

        if noise_cov is None:
            raise ValueError('LinearGaussianModel requires a `noise_cov`.')

        super().__init__(model=model, noise_cov=noise_cov)

    # Delegating properties so KalmanPredict / RTS smoother can read mat_x / mat_u
    # without knowing the inner model layout.
    @property
    def mat_x(
        self
    ) -> Tensor:
        return self.model.mat_x

    @property
    def mat_u(
        self
    ) -> Tensor | None:
        return self.model.mat_u


class NonlinearAdditiveModel(AdditiveModel):
    """Nonlinear deterministic dynamics with additive Gaussian noise."""

    def __init__(
        self,
        f : Callable[[Tensor, Tensor], Tensor] | torch.nn.Module | None = None,
        noise_cov : Tensor | PSDMatrix | None = None,
        in_dim : int | None = None,
        out_dim : int | None = None,
        model : NonlinearModel | None = None,
    ):
        
        if model is None:
            if f is None or in_dim is None or out_dim is None:
                raise ValueError('Provide either `model` or all of (`f`, `in_dim`, `out_dim`).')
            model = NonlinearModel(f=f, in_dim=in_dim, out_dim=out_dim)
        elif not isinstance(model, NonlinearModel):
            raise ValueError(f'`model` must be a NonlinearModel for NonlinearAdditiveModel, got {type(model)}.')

        if noise_cov is None:
            raise ValueError('NonlinearAdditiveModel requires a `noise_cov`.')

        super().__init__(model=model, noise_cov=noise_cov)
