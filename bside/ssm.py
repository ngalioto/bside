import torch
from torch import Tensor
from bside.dynamics import AdditiveModel, Model, IdentityModel
from typing import Tuple


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
    
    def _rollout(
        self,
        x : Tensor,
        u : Tensor | None,
        T : int,
        keep_y0 : bool,
        process_noise : bool
    ) -> Tensor:

        if process_noise and not isinstance(self.dynamics, AdditiveModel):
            raise ValueError(
                '`process_noise=True` needs dynamics with a noise model (an AdditiveModel such as '
                f'LinearGaussianModel), got {type(self.dynamics).__name__}.'
            )
        return self.dynamics.predict(x, u, T, keep_x0=keep_y0, stochastic=process_noise)

    def predict(
        self,
        x : Tensor,
        u : Tensor = None,
        T : int = 1,
        return_x : bool = False,
        keep_y0 : bool = False,
        process_noise : bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:

        """
        Simulate the model for `T` steps from state `x` and return the observations
        **without measurement noise**, y = h(x).

        With ``process_noise=False`` (default) the dynamics run deterministically:
        the model's best-guess trajectory, e.g. for using a fitted model whose
        process noise only stands for model error.  With ``process_noise=True``
        each step samples the dynamics' noise, giving one realization of the latent
        trajectory under that uncertainty.
        """

        x = self._rollout(x, u, T, keep_y0, process_noise)
        y = self.observations(x)

        return y if not return_x else (x, y)

    def measure(
        self,
        x : Tensor,
        u : Tensor = None,
        T : int = 1,
        return_x : bool = False,
        keep_y0 : bool = False,
        process_noise : bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:

        """
        Simulate the model for `T` steps from state `x` and return noisy
        measurements **with measurement noise**, y = h(x) + eta.

        With ``process_noise=False`` (default) the dynamics run deterministically,
        as when generating data from a deterministic ground truth.  With
        ``process_noise=True`` each step also samples the dynamics' noise: a full
        draw from the stochastic model, including model uncertainty.
        """

        x = self._rollout(x, u, T, keep_y0, process_noise)
        y = self.observations.sample(x)  # independent noise for every point (and trajectory)

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
