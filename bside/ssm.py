import torch
from torch import Tensor
from bside.dynamics import Model, IdentityModel
from bside.dataset import Data, DataTrajectories
from typing import Union, Tuple, Callable


class SSM(torch.nn.Module):

    """
    A state-space model (SSM) defined by a dynamics model and an observation
    model.  Dynamics maps the state vector to the next state; observations maps
    the state vector to the measurement.  Both models can be linear, nonlinear,
    Gaussian, or arbitrary-noise; the SSM only requires that they implement the
    `Model` interface.

    Encoder-style models (those that map a window of past observations and
    inputs to a latent initial state) are no longer part of `SSM`; see
    `bside.subspace_encoder.SubspaceEncoder` for that variant.
    """

    def __init__(
        self,
        xdim : int,
        ydim : int,
        dynamics : Model,
        observations : Model = None
    ):
        
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim

        if observations is None:
            if xdim != ydim:
                raise ValueError(
                    f"An observation model must be provided when xdim ({xdim}) != ydim ({ydim})."
                )
            observations = IdentityModel(self.xdim)

        self.dynamics = dynamics
        self.observations = observations
    
    def __repr__(
        self
    ) -> str:
        
        return (
            "State-space model (SSM):\n"
            f"  State dimension: {self.xdim}\n"
            f"  Output dimension: {self.ydim}\n"
            f"  Dynamics:\n    {self.dynamics}\n"
            f"  Observations:\n    {self.observations}\n"
        )
    
    def predict(
        self,
        x : Tensor,
        u : Tensor = None,
        T : int = 1,
        return_x : bool = False,
        keep_y0 : bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        
        x = self.dynamics.predict(x, u, T, keep_x0=keep_y0)
        y = self.observations(x)

        return y if not return_x else (x, y)
    
    def measure(
        self,
        x : Tensor,
        u : Tensor = None,
        T : int = 1,
        return_x : bool = False,
        keep_y0 : bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        
        x = self.dynamics.predict(x, u, T, keep_x0=keep_y0)
        y = self.observations.sample(x, N=(T + keep_y0))

        return y if not return_x else (x, y)

    def update(
        self,
        params : Tensor | None = None
    ) -> None:
        
        """
        Propagate the flat parameter vector ``params`` into both submodels'
        learnable matrices.  When ``params`` is None each submodel uses its
        own internal `nn.Parameter`.
        """

        self.dynamics.update(params)
        self.observations.update(params)
