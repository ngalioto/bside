import torch
from torch import Tensor
import copy
from typing import Tuple, List
from abc import ABC, abstractmethod

from bside.ssm import SSM
from bside.dynamics import Model, LinearGaussianModel, IdentityModel, LinearModel
from bside.models import PSDMatrix

from bside.dataset import Data
from bside.filtering import FilteringDistribution, functional as F


def _ensure_linear_gaussian(
    model: Model,
    dim: int,
    eps: float = 1e-12
) -> LinearGaussianModel:
    """
    Wrap an `IdentityModel` (or any `LinearModel` with no noise) in a
    `LinearGaussianModel` with negligible (epsilon * I) observation noise so the
    Kalman filter can consume it uniformly. Models that are already linear-Gaussian
    are returned unchanged.
    """

    if isinstance(model, LinearGaussianModel):
        return model
    if isinstance(model, LinearModel):
        return LinearGaussianModel(
            model=model,
            noise_cov=PSDMatrix(eps * torch.eye(dim))
        )
    raise ValueError(
        f"Kalman filter requires a LinearModel (or subclass); got {type(model)}."
    )

"""
TODO: Need to add particle filter and various Gaussian quadratures

TODO: Build a off-the-shelf filters like Kalman, Unscented, Gauss-Hermite, Particle...
"""
    
class FilterPredict(ABC):

    """
    Defines the structure of the predict function in a state estimation filter.

    All concrete predict implementations push a `FilteringDistribution` through a
    `Model`, optionally returning the cross-covariance between the input and output
    distributions for use in the corresponding update step.
    """

    @abstractmethod
    def __call__(
        self,
        model: Model,
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
        
        super().__init__()
        self.lmbda = lmbda

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        if dist.particles is None:
            dist.form_ut_points(self.lmbda)
        return F.gaussian_quadrature(model, dist, u, crossCov)
    
class GaussHermitePredict(FilterPredict):

    def __init__(
        self
    ) -> None:
        
        super().__init__()

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        if dist.particles is None:
            dist.form_gh_points()
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
    
class GaussQuadKalmanUpdate(FilterUpdate):

    def __call__(
        self,
        y: Tensor,
        dist_x: FilteringDistribution,
        dist_y: FilteringDistribution,
        U: Tensor,
        Sinv: Tensor | None = None
    ) -> FilteringDistribution:
        
        # We can keep the weights, but the quadrature points will be outdated
        dist_x.particles = None
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
        model: SSM,
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
        return_history: bool = False,
        compute_log_prob: bool = False
    ) -> FilteringDistribution | List[FilteringDistribution]:
        
        T = len(data)
        self.dist = copy.copy(init_dist)

        if return_history:
            state_estimates = [copy.deepcopy(self.dist)]

        if compute_log_prob:
            log_prob = 0.0
        
        if y0:
            t = 0
            y_dist, U = self.observations_filter(
                model=self.model.observations,
                dist=self.dist,
                u=data.u[t] if data.u is not None else None,
                crossCov=True
            )
            
            if compute_log_prob:
                log_prob += y_dist.log_prob(data.y[t])

            self.dist = self.update(data.y[t], self.dist, y_dist, U)

            if return_history:
                state_estimates.append(copy.deepcopy(self.dist))

        for t in range(1 if y0 else 0, T):
            self.dist = self.dynamics_filter(
                model=self.model.dynamics,
                dist=self.dist,
                u=data.u[t-1] if data.u is not None else None,
                crossCov=False
            )

            y_dist, U = self.observations_filter(
                model=self.model.observations,
                dist=self.dist,
                u=data.u[t] if data.u is not None else None,
                crossCov=True
            )

            if compute_log_prob:
                log_prob += y_dist.log_prob(data.y[t])

            self.dist = self.update(data.y[t], self.dist, y_dist, U)
            if return_history:
                state_estimates.append(copy.deepcopy(self.dist))

        output = state_estimates if return_history else self.dist
        return (output, log_prob) if compute_log_prob else output
    
    def nlog_marginal_likelihood(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        params: Tensor | None = None
    ) -> Tensor:
        
        if params is not None:
            self.model.update(params)
        
        _, logprob = self.filter(
            data = data, 
            init_dist = init_dist, 
            y0 = y0,
            return_history=False,
            compute_log_prob=True
        )

        return -logprob
    

class KalmanFilter(Filter):

    def __init__(
        self,
        model: SSM
    ) -> None:
        
        if not isinstance(model.dynamics, LinearGaussianModel):
            raise ValueError(f"Kalman filter requires a linear Gaussian dynamics model, but got type {type(model.dynamics)}")
        # IdentityModel / LinearModel observations are accepted and wrapped with a
        # near-zero noise covariance so the same Kalman update math applies.
        model.observations = _ensure_linear_gaussian(model.observations, model.ydim)

        super().__init__(
            model = model,
            dynamics_filter = KalmanPredict(),
            observations_filter = KalmanPredict(),
            update = KalmanUpdate()
        )

class UnscentedKalmanFilter(Filter):

    def __init__(
        self,
        model: SSM,
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
            update = GaussQuadKalmanUpdate()
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False
    ) -> FilteringDistribution | List[FilteringDistribution]:
    
        init_dist = copy.copy(init_dist)
        init_dist.form_ut_weights(
            alpha = self.alpha,
            beta = self.beta,
            kappa = self.kappa,
            lmbda = self.lmbda
        )

        return super().filter(data, init_dist, y0, return_history, compute_log_prob)
    
class GaussHermiteFilter(Filter):

    def __init__(
        self,
        model: SSM,
        order: int = 3
    ) -> None:
        
        self.order = order

        super().__init__(
            model = model,
            dynamics_filter = GaussHermitePredict(),
            observations_filter = GaussHermitePredict(),
            update = GaussQuadKalmanUpdate()
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False
    ) -> FilteringDistribution | List[FilteringDistribution]:

        init_dist = copy.copy(init_dist)
        init_dist.form_gh_weights(
            order = self.order
        )

        return super().filter(data, init_dist, y0, return_history, compute_log_prob)

class EnsembleKalmanFilter(Filter):

    def __init__(
        self,
        model: SSM,
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
        return_history: bool = False,
        compute_log_prob: bool = False
    ) -> FilteringDistribution | List[FilteringDistribution]:
    
        init_dist = copy.copy(init_dist)
        init_dist.sample_particles(self.ensemble_size)

        return super().filter(
            data=data,
            init_dist=init_dist,
            y0=y0,
            return_history=return_history,
            compute_log_prob=compute_log_prob,
        )