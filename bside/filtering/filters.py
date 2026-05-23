import torch
from torch import Tensor
import copy
from typing import Tuple, List, Callable
from abc import ABC, abstractmethod

from bside.ssm import SSM
from bside.dynamics import Model, AdditiveModel, LinearGaussianModel, LinearModel
from bside.models import PSDMatrix

from bside.dataset import Data
from bside.filtering import FilteringDistribution, functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_linear_gaussian(
    model: Model,
    dim: int,
    eps: float = 1e-12
) -> LinearGaussianModel:
    """
    Wrap an `IdentityModel` (or any `LinearModel` with no noise) in a
    `LinearGaussianModel` with negligible (``eps * I``) observation noise so the
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


# ---------------------------------------------------------------------------
# Predict primitives (one per algorithm family)
# ---------------------------------------------------------------------------

class FilterPredict(ABC):
    """
    A `FilterPredict` pushes a `FilteringDistribution` through a `Model` and
    optionally returns the input-output cross-covariance needed by the update.
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
    """Closed-form linear-Gaussian predict (used by KalmanFilter)."""

    def __call__(
        self,
        model: LinearGaussianModel,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        return F.kf_predict(model, dist, u, crossCov)


class SquareRootKalmanPredict(FilterPredict):
    """
    Square-root linear-Gaussian predict.  Operates entirely on the Cholesky
    factor of the covariance via QR of ``[A L_x | L_Q]`` (Park & Kailath).
    """

    def __call__(
        self,
        model: LinearGaussianModel,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        return F.srkf_predict(model, dist, u, crossCov)


class EnsembleKalmanPredict(FilterPredict):
    """Ensemble Kalman predict: propagate particles + Gaussian noise."""

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        return F.enkf_predict(model, dist, u, crossCov)


class UnscentedKalmanPredict(FilterPredict):
    """
    Unscented transform predict.  If ``regenerate_points=True`` (default), the
    sigma points are re-generated around the input distribution's (mean, cov)
    every call; otherwise the previously propagated particles are re-used
    (faster, slightly less accurate -- mirrors the previous behaviour as an
    opt-in optimization).
    """

    def __init__(
        self,
        lmbda: float = 1.0,
        regenerate_points: bool = True,
    ) -> None:
        
        super().__init__()
        self.lmbda = lmbda
        self.regenerate_points = regenerate_points

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        if self.regenerate_points or dist.particles is None:
            dist.form_ut_points(self.lmbda)
        return F.gaussian_quadrature(model, dist, u, crossCov)


class GaussHermitePredict(FilterPredict):
    """Gauss-Hermite quadrature predict (tensor-product nodes)."""

    def __init__(
        self,
        regenerate_points: bool = True,
    ) -> None:
        
        super().__init__()
        self.regenerate_points = regenerate_points

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        
        if self.regenerate_points or dist.particles is None:
            dist.form_gh_points()
        return F.gaussian_quadrature(model, dist, u, crossCov)


class CubatureKalmanPredict(FilterPredict):
    """3rd-order spherical-cubature predict (Arasaratnam & Haykin, 2009)."""

    def __init__(
        self,
        regenerate_points: bool = True,
    ) -> None:
        
        super().__init__()
        self.regenerate_points = regenerate_points

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        if self.regenerate_points or dist.particles is None:
            dist.form_cubature_points()
        return F.gaussian_quadrature(model, dist, u, crossCov)


# ---------------------------------------------------------------------------
# Update primitives
# ---------------------------------------------------------------------------

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
    """
    Kalman update for deterministic-quadrature filters.  Clears the stale
    particle support so the next predict regenerates fresh sigma / cubature
    points around the updated (mean, cov).
    """

    def __call__(
        self,
        y: Tensor,
        dist_x: FilteringDistribution,
        dist_y: FilteringDistribution,
        U: Tensor,
        Sinv: Tensor | None = None
    ) -> FilteringDistribution:
        
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


# ---------------------------------------------------------------------------
# Base recursion shared by all Gaussian filters
# ---------------------------------------------------------------------------

class Filter(ABC):
    """
    A Filter is a (dynamics predict, observations predict, update) triple plus
    a recursive `filter(...)` driver.  All concrete Gaussian filters share the
    same predict/observe/update loop -- they differ only in which predict /
    update primitives they wire up.

    The base recursion plumbs ``compute_log_prob`` and an optional ``params``
    tensor through every concrete filter so the marginal-likelihood path is
    uniform across Kalman / Unscented / Gauss-Hermite / Cubature / Ensemble.
    """

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
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution]:
        
        if params is not None:
            self.model.update(params)
        
        T = len(data)
        # Deepcopy init_dist so that updates to self.dist (mean/cov in-place via
        # the kalman_update setter path) do not mutate the user's init_dist.
        self.dist = copy.deepcopy(init_dist)

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
                log_prob = log_prob + y_dist.log_prob(data.y[t])

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
                log_prob = log_prob + y_dist.log_prob(data.y[t])

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
        
        _, logprob = self.filter(
            data=data,
            init_dist=init_dist,
            y0=y0,
            return_history=False,
            compute_log_prob=True,
            params=params,
        )

        return -logprob


# ---------------------------------------------------------------------------
# Concrete Gaussian filters
# ---------------------------------------------------------------------------

class KalmanFilter(Filter):
    """Standard linear-Gaussian Kalman filter."""

    def __init__(
        self,
        model: SSM
    ) -> None:
        
        if not isinstance(model.dynamics, LinearGaussianModel):
            raise ValueError(
                f"Kalman filter requires a linear-Gaussian dynamics model, got {type(model.dynamics)}."
            )
        # Accept IdentityModel/LinearModel observations by wrapping with epsilon noise.
        model.observations = _ensure_linear_gaussian(model.observations, model.ydim)

        super().__init__(
            model=model,
            dynamics_filter=KalmanPredict(),
            observations_filter=KalmanPredict(),
            update=KalmanUpdate(),
        )


class SquareRootKalmanFilter(Filter):
    """
    Square-root Kalman filter.  Same fixed-point as the standard KF but never
    materializes the dense covariance: predict and update operate on the
    Cholesky factor through QR factorizations.  Recommended when the state
    covariance is ill-conditioned.
    """

    def __init__(
        self,
        model: SSM
    ) -> None:

        if not isinstance(model.dynamics, LinearGaussianModel):
            raise ValueError(
                f"SquareRootKalmanFilter requires a linear-Gaussian dynamics model, got {type(model.dynamics)}."
            )
        model.observations = _ensure_linear_gaussian(model.observations, model.ydim)

        super().__init__(
            model=model,
            dynamics_filter=SquareRootKalmanPredict(),
            observations_filter=None,  # update fuses predict+update for the obs step
            update=None,
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution]:
        """
        Custom recursion that pairs the square-root predict with the square-root
        update (the latter folds the observation prediction into a single QR).
        """

        if params is not None:
            self.model.update(params)

        T = len(data)
        self.dist = copy.deepcopy(init_dist)
        if return_history:
            state_estimates = [copy.deepcopy(self.dist)]

        if compute_log_prob:
            log_prob = 0.0

        if y0:
            t = 0
            u_t = data.u[t] if data.u is not None else None
            if compute_log_prob:
                y_dist = F.kf_predict(self.model.observations, self.dist, u=u_t, crossCov=False)
                log_prob = log_prob + y_dist.log_prob(data.y[t])
            self.dist = F.srkf_update(data.y[t], self.dist, self.model.observations, u=u_t)
            if return_history:
                state_estimates.append(copy.deepcopy(self.dist))

        for t in range(1 if y0 else 0, T):
            self.dist = self.dynamics_filter(
                model=self.model.dynamics,
                dist=self.dist,
                u=data.u[t-1] if data.u is not None else None,
                crossCov=False,
            )

            u_t = data.u[t] if data.u is not None else None
            if compute_log_prob:
                # For the log-likelihood we need the *predicted* observation Gaussian.
                y_dist = F.kf_predict(self.model.observations, self.dist, u=u_t, crossCov=False)
                log_prob = log_prob + y_dist.log_prob(data.y[t])

            self.dist = F.srkf_update(data.y[t], self.dist, self.model.observations, u=u_t)
            if return_history:
                state_estimates.append(copy.deepcopy(self.dist))

        output = state_estimates if return_history else self.dist
        return (output, log_prob) if compute_log_prob else output


class UnscentedKalmanFilter(Filter):

    def __init__(
        self,
        model: SSM,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        regenerate_points: bool = True,
    ) -> None:
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        lmbda = alpha**2 * (model.xdim + kappa) - model.xdim
        self.lmbda = lmbda

        super().__init__(
            model=model,
            dynamics_filter=UnscentedKalmanPredict(lmbda, regenerate_points=regenerate_points),
            observations_filter=UnscentedKalmanPredict(lmbda, regenerate_points=regenerate_points),
            update=GaussQuadKalmanUpdate(),
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution]:
    
        init_dist = copy.copy(init_dist)
        init_dist.form_ut_weights(
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            lmbda=self.lmbda,
        )

        return super().filter(data, init_dist, y0, return_history, compute_log_prob, params)


class GaussHermiteFilter(Filter):

    def __init__(
        self,
        model: SSM,
        order: int = 3,
        regenerate_points: bool = True,
    ) -> None:
        
        self.order = order

        super().__init__(
            model=model,
            dynamics_filter=GaussHermitePredict(regenerate_points=regenerate_points),
            observations_filter=GaussHermitePredict(regenerate_points=regenerate_points),
            update=GaussQuadKalmanUpdate(),
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution]:

        init_dist = copy.copy(init_dist)
        init_dist.form_gh_weights(order=self.order)

        return super().filter(data, init_dist, y0, return_history, compute_log_prob, params)


class CubatureKalmanFilter(Filter):
    """
    Cubature Kalman filter (Arasaratnam & Haykin, 2009).  Uses 2n
    spherical 3rd-order cubature points with equal weights -- cheaper than
    Gauss-Hermite for moderate dimensions, often comparable accuracy.
    """

    def __init__(
        self,
        model: SSM,
        regenerate_points: bool = True,
    ) -> None:
        
        super().__init__(
            model=model,
            dynamics_filter=CubatureKalmanPredict(regenerate_points=regenerate_points),
            observations_filter=CubatureKalmanPredict(regenerate_points=regenerate_points),
            update=GaussQuadKalmanUpdate(),
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution]:

        init_dist = copy.copy(init_dist)
        init_dist.form_cubature_weights()

        return super().filter(data, init_dist, y0, return_history, compute_log_prob, params)


class EnsembleKalmanFilter(Filter):

    def __init__(
        self,
        model: SSM,
        ensemble_size: int
    ) -> None:
        
        self.ensemble_size = ensemble_size
        
        super().__init__(
            model=model,
            dynamics_filter=EnsembleKalmanPredict(),
            observations_filter=EnsembleKalmanPredict(),
            update=EnsembleKalmanUpdate(),
        )

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution]:
    
        init_dist = copy.copy(init_dist)
        init_dist.sample_particles(self.ensemble_size)

        return super().filter(
            data=data,
            init_dist=init_dist,
            y0=y0,
            return_history=return_history,
            compute_log_prob=compute_log_prob,
            params=params,
        )


# ---------------------------------------------------------------------------
# Particle filter (bootstrap with optional ESS-based resampling)
# ---------------------------------------------------------------------------

class ParticleFilter:
    """
    Bootstrap particle filter with effective-sample-size triggered resampling.

    The observation model must be an `AdditiveModel` (we use its Cholesky factor
    to evaluate ``log p(y | x_i)`` without forming the inverse).  Dynamics can
    be any `Model` -- they are propagated via ``model.sample`` so arbitrary
    non-Gaussian process noise is supported.
    """

    def __init__(
        self,
        model: SSM,
        n_particles: int,
        resample_method: str = 'systematic',
        ess_threshold: float = 0.5,
    ) -> None:
        
        if not isinstance(model.observations, AdditiveModel):
            raise ValueError(
                "ParticleFilter requires an additive-Gaussian observation model so the "
                "per-particle likelihood log p(y|x) is tractable."
            )

        self.model = model
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.ess_threshold = ess_threshold

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution]:

        if params is not None:
            self.model.update(params)

        T = len(data)
        dist = copy.copy(init_dist)
        dist.sample_particles(self.n_particles)

        if return_history:
            history = [copy.deepcopy(dist)]
        if compute_log_prob:
            log_prob = 0.0

        if y0:
            t = 0
            dist, lm = F.particle_filter_update(
                data.y[t], dist, self.model.observations,
                resample_threshold=self.ess_threshold,
                resample_method=self.resample_method,
            )
            if compute_log_prob:
                log_prob = log_prob + lm
            if return_history:
                history.append(copy.deepcopy(dist))

        for t in range(1 if y0 else 0, T):
            dist = F.bootstrap_pf_predict(
                self.model.dynamics, dist,
                u=data.u[t-1] if data.u is not None else None,
            )
            dist, lm = F.particle_filter_update(
                data.y[t], dist, self.model.observations,
                resample_threshold=self.ess_threshold,
                resample_method=self.resample_method,
            )
            if compute_log_prob:
                log_prob = log_prob + lm
            if return_history:
                history.append(copy.deepcopy(dist))

        self.dist = dist
        output = history if return_history else dist
        return (output, log_prob) if compute_log_prob else output

    def nlog_marginal_likelihood(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        params: Tensor | None = None,
    ) -> Tensor:
        _, logprob = self.filter(
            data=data,
            init_dist=init_dist,
            y0=y0,
            return_history=False,
            compute_log_prob=True,
            params=params,
        )
        return -logprob
