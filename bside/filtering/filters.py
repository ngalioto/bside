import torch
from torch import Tensor
import copy
from typing import Tuple, List
from abc import ABC, abstractmethod

from bside.dynamics import Model, LinearGaussianModel

# this is why Filter is in its own file. Otherwise, we would have a circular import since SSM imports FilteringDistribution
from bside.state_space import SSM, HMM

from bside.dataset import Data
from bside.filtering.distributions import FilteringDistribution
from bside.filtering import functional as F

"""
TODO: Need to add particle filter and various Gaussian quadratures

TODO: Build a off-the-shelf filters like Kalman, Unscented, Gauss-Hermite, Particle...
"""
    
class FilterPredict(ABC):

    """
    Defines the structure of the predict function in a state estimation filter.

    Could initialize with the model, but not sure if that adds any benefit.
    """

    @abstractmethod
    def __call__(
        self,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        pass

class KalmanPredict(FilterPredict):

    def __call__(
        self,
        model: LinearGaussianModel,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        return F.kf_predict(model, dist, u, crossCov)
    
class EnsembleKalmanPredict(FilterPredict):

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        return F.enkf_predict(model, dist, u, crossCov)
    
class UnscentedKalmanPredict(FilterPredict):

    def __init__(
        self,
        lmbda: float = 1.0,
    ) -> None:
        
        self.lmbda = lmbda

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        dist.form_ut_points(self.lmbda)
        return F.gaussian_quadrature(model, dist, u, crossCov)

class FilterUpdate(ABC):

    @abstractmethod
    def __call__(
        self,
        y: Tensor,
        dist_x: FilteringDistribution,
        dist_y: FilteringDistribution,
        U: Tensor,
        Sinv: Tensor | None = None
    ) -> FilteringDistribution:
        
        pass

class KalmanUpdate(FilterUpdate):

    def __call__(
        self,
        y: Tensor,
        dist_x: FilteringDistribution,
        dist_y: FilteringDistribution,
        U: Tensor,
        Sinv: Tensor | None = None
    ) -> FilteringDistribution:
        
        return F.kalman_update(y, dist_x, dist_y, U, Sinv)
    
class EnsembleKalmanUpdate(FilterUpdate):

    def __call__(
        self,
        y: Tensor,
        dist_x: FilteringDistribution,
        dist_y: FilteringDistribution,
        U: Tensor,
        Sinv: Tensor | None = None
    ) -> FilteringDistribution:
        
        return F.enkf_update(y, dist_x, dist_y, U, Sinv)

class Filter(ABC):

    def __init__(
        self,
        model: HMM,
        dynamics_filter: FilterPredict | None = None,
        observations_filter: FilterPredict | None = None,
        update: FilterUpdate | None = None
    ) -> None:
        
        self.model = model
        self.dynamics_filter = dynamics_filter
        self.observations_filter = observations_filter
        self.update = update


    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        obs_freq: int | Tensor | None = None,
        return_history: bool = False
    ) -> FilteringDistribution | List[FilteringDistribution]:
        
        if obs_freq is None:
            obs_freq = torch.ones(1, dtype=torch.int) if obs_freq is None else obs_freq
        elif isinstance(obs_freq, int):
            obs_freq = obs_freq * torch.ones(1, dtype=torch.int)
        T = len(data)
        self.dist = init_dist

        if return_history:
            state_estimates = []

        # log_prob = 0.0
        if y0:
            y_dist, U = self.observations_filter(
                self.model,
                self.dist, 
                data.u[t] if data.u is not None else None,
                crossCov=True
            )
            # log_prob += y_dist.log_prob(data.y[t])
            self.dist = self.update(data.y[t], self.dist, y_dist, U)
            if return_history:
                state_estimates.append(copy.deepcopy(self.dist))

        # obs_freq = obs_freq.expand(T) if len(obs_freq) == 1 else obs_freq

        for t in range(1 if y0 else 0, T):
            self.dist = self.dynamics_filter(data.u[t] if data.u is not None else None)
            y_dist, U = self.observations_filter(
                self.dist, 
                data.u[t] if data.u is not None else None,
                crossCov=True
            )
            # log_prob += y_dist.log_prob(data.y[t])
            self.dist = self.update(data.y[t], self.dist, y_dist, U)
            if return_history:
                state_estimates.append(copy.deepcopy(self.dist))

        return state_estimates if return_history else self.dist
        # return (state_estimates if return_history else self.dist, log_prob)
    

class KalmanFilter(Filter):

    def __init__(
        self,
        model: HMM
    ) -> None:
        
        if not isinstance(model.dynamics, LinearGaussianModel):
            raise ValueError(f"Kalman filter requires a linear Gaussian model, but {type(model.dynamics)=}")
        if not isinstance(model.observations, LinearGaussianModel):
            raise ValueError(f"Kalman filter requires a linear Gaussian model, but {type(model.observations)=}")
        
        super().__init__(
            model = model,
            dynamics_filter = KalmanPredict(),
            observations_filter = KalmanPredict(),
            update = KalmanUpdate()
        )

class UnscentedKalmanFilter(Filter):

    def __init__(
        self,
        model: HMM,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0
    ) -> None:
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        lmbda = alpha**2 * (model.xdim + kappa) - model.xdim
        self.lmbda = lmbda

        super().__init__(
            model = model,
            dynamics_filter = UnscentedKalmanPredict(lmbda),
            observations_filter = UnscentedKalmanPredict(lmbda),
            update = KalmanUpdate()
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        obs_freq: int | Tensor | None = None,
        return_history: bool = False
    ) -> FilteringDistribution | List[FilteringDistribution]:
    
        init_dist.form_ut_weights(
            n = self.model.xdim,
            alpha = self.alpha,
            beta = self.beta,
            kappa = self.kappa
        )

        return super().filter(data, init_dist, y0, obs_freq, return_history)

class EnsembleKalmanFilter(Filter):

    def __init__(
        self,
        model: HMM,
        ensemble_size: int
    ) -> None:
        
        self.ensemble_size = ensemble_size
        
        super().__init__(
            model = model,
            dynamics_filter = EnsembleKalmanPredict(),
            observations_filter = EnsembleKalmanPredict(),
            update = EnsembleKalmanUpdate()
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        obs_freq: int | Tensor | None = None,
        return_history: bool = False
    ) -> FilteringDistribution | List[FilteringDistribution]:
    
        init_dist.sample_particles(self.ensemble_size)

        return super().filter(data, init_dist, y0, obs_freq, return_history)