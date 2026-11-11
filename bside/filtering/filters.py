"""
Bayesian filters.

A filter is assembled from the two halves of the state-space model:

* a **predict** unit that pushes the state distribution through the dynamics
  model (deterministic map + process noise), and
* an **update** unit that pushes the predicted state through the observation
  model (deterministic map + measurement noise) to obtain
  ``p(y_k | y_{1:k-1})`` -- the marginal-likelihood term -- and then conditions
  on ``y_k``.

`Filter.filter` is the single recursion shared by every filter; concrete
filters only choose which predict / update primitives to plug in (and, for the
square-root and particle filters, override `update_step` / `predict_step`).
The marginal likelihood ``p(y_{1:n} | theta) = prod_k p(y_k | y_{1:k-1}, theta)``
accumulates along the way, which is what `bside.sysid.Posterior` consumes.

Time convention (shared with `SSM.measure`): with ``y0=False`` data row ``t``
holds ``y_{t+1}`` and the input ``u_t`` that drove ``x_t -> x_{t+1}``; with
``y0=True`` row ``t`` holds ``y_t`` (the first row observes ``x_0``) and the
step ``x_{t-1} -> x_t`` uses ``u_{t-1}``.  Either way the returned history
satisfies ``history[k] = p(x_k | y_{...k})``, i.e. index ``k`` is state ``x_k``.
"""

from abc import ABC
from typing import Tuple, List

import torch
from torch import Tensor

from bside.ssm import SSM
from bside.dynamics import Model, AdditiveModel, LinearGaussianModel, LinearModel
from bside.models import PSDMatrix

from bside.dataset import Data
from bside.filtering.distributions import FilteringDistribution
from bside.filtering import functional as F


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
    are returned unchanged.  The wrapper shares the inner model, so parameter
    updates on the original model are seen by the filter.
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


def _input_at(u: Tensor | None, t: int) -> Tensor | None:
    """Input at time row ``t`` of a ``(..., T, udim)`` tensor (None-safe)."""

    return None if u is None else u[..., t, :]


# ---------------------------------------------------------------------------
# Predict primitives (one per approximation family)
# ---------------------------------------------------------------------------

class FilterPredict(ABC):
    """
    A `FilterPredict` pushes a `FilteringDistribution` through a `Model` and
    optionally returns the input-output cross-covariance needed by the update.
    """

    def __call__(
        self,
        model: Model,
        dist: FilteringDistribution,
        u: Tensor = None,
        crossCov: bool = False
    ) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
        raise NotImplementedError


class KalmanPredict(FilterPredict):
    """Closed-form linear-Gaussian predict (used by KalmanFilter)."""

    def __call__(self, model, dist, u=None, crossCov=False):
        return F.kf_predict(model, dist, u, crossCov)


class SquareRootKalmanPredict(FilterPredict):
    """
    Square-root linear-Gaussian predict.  Operates entirely on the Cholesky
    factor of the covariance via QR of ``[A L_x | L_Q]`` (Park & Kailath).
    """

    def __call__(self, model, dist, u=None, crossCov=False):
        return F.srkf_predict(model, dist, u, crossCov)


class EnsembleKalmanPredict(FilterPredict):
    """Ensemble Kalman predict: propagate particles + Gaussian noise."""

    def __call__(self, model, dist, u=None, crossCov=False):
        return F.enkf_predict(model, dist, u, crossCov)


class _QuadraturePredict(FilterPredict):
    """
    Deterministic-quadrature predict.  If ``regenerate_points=True`` (default)
    the points are re-formed around the input distribution's (mean, cov) on
    every call; otherwise previously propagated points are re-used when
    present (faster, less accurate).
    """

    def __init__(self, regenerate_points: bool = True) -> None:
        self.regenerate_points = regenerate_points

    def form_points(self, dist: FilteringDistribution) -> None:
        raise NotImplementedError

    def __call__(self, model, dist, u=None, crossCov=False):
        if self.regenerate_points or dist.particles is None:
            self.form_points(dist)
        return F.gaussian_quadrature(model, dist, u, crossCov)


class UnscentedKalmanPredict(_QuadraturePredict):
    """Unscented-transform predict (2n+1 sigma points)."""

    def __init__(self, lmbda: float = 1.0, regenerate_points: bool = True) -> None:
        super().__init__(regenerate_points)
        self.lmbda = lmbda

    def form_points(self, dist):
        dist.form_ut_points(self.lmbda)


class GaussHermitePredict(_QuadraturePredict):
    """Gauss-Hermite quadrature predict (tensor-product nodes)."""

    def form_points(self, dist):
        dist.form_gh_points()


class CubatureKalmanPredict(_QuadraturePredict):
    """3rd-order spherical-cubature predict (Arasaratnam & Haykin, 2009)."""

    def form_points(self, dist):
        dist.form_cubature_points()


# ---------------------------------------------------------------------------
# Update primitives
# ---------------------------------------------------------------------------

class FilterUpdate(ABC):
    """
    Condition the predicted state on ``y`` given the predicted observation
    distribution ``dist_y`` and the state-observation cross-covariance ``U``.
    """

    def __call__(
        self,
        y: Tensor,
        dist_x: FilteringDistribution,
        dist_y: FilteringDistribution,
        U: Tensor,
        Sinv: Tensor | None = None
    ) -> FilteringDistribution:
        raise NotImplementedError


class KalmanUpdate(FilterUpdate):
    """Gaussian conditioning ``m + K (y - mu_y)``, ``P - K U^T``."""

    def __call__(self, y, dist_x, dist_y, U, Sinv=None):
        return F.kalman_update(y, dist_x, dist_y, U, Sinv)


class GaussQuadKalmanUpdate(KalmanUpdate):
    """
    Kalman update for deterministic-quadrature filters.  The posterior carries
    the quadrature weights but no points, so the next predict forms fresh
    points around the updated (mean, cov).
    """


class EnsembleKalmanUpdate(FilterUpdate):
    """Stochastic (perturbed-observation) ensemble update."""

    def __call__(self, y, dist_x, dist_y, U, Sinv=None):
        return F.enkf_update(y, dist_x, dist_y, U, Sinv)


# ---------------------------------------------------------------------------
# Base recursion shared by all filters
# ---------------------------------------------------------------------------

class Filter(ABC):
    """
    A Filter is a (dynamics predict, observations predict, update) triple plus
    the recursive `filter(...)` driver.

    Subclasses customize the recursion only through three hooks:

    * `initialize(init_dist)` -- prepare the prior ``p(x_0)`` (e.g. attach
      quadrature weights or draw particles)
    * `predict_step(dist, u)` -- ``p(x_{k-1} | ...) -> p(x_k | ...)`` through
      the dynamics model
    * `update_step(y, dist, u, compute_log_prob)` -- through the observation
      model, returning the posterior and ``log p(y_k | y_{1:k-1})``
    """

    #: Whether the filter supports a leading batch of trajectories in ``Data``.
    supports_batch: bool = True

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

    # -- model halves -----------------------------------------------------

    @property
    def dynamics(self) -> Model:
        return self.model.dynamics

    @property
    def observations(self) -> Model:
        return self.model.observations

    # -- hooks ------------------------------------------------------------

    def initialize(
        self,
        init_dist: FilteringDistribution
    ) -> FilteringDistribution:
        """Return a private copy of the prior so the caller's object is untouched."""

        return init_dist.snapshot()

    def predict_step(
        self,
        dist: FilteringDistribution,
        u: Tensor | None = None
    ) -> FilteringDistribution:

        return self.dynamics_filter(model=self.dynamics, dist=dist, u=u, crossCov=False)

    def update_step(
        self,
        y: Tensor,
        dist: FilteringDistribution,
        u: Tensor | None = None,
        compute_log_prob: bool = False
    ) -> Tuple[FilteringDistribution, Tensor | None]:

        y_dist, U = self.observations_filter(model=self.observations, dist=dist, u=u, crossCov=True)
        log_lik = y_dist.log_prob(y) if compute_log_prob else None
        return self.update(y, dist, y_dist, U), log_lik

    # -- recursion --------------------------------------------------------

    def filter(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        return_history: bool = False,
        compute_log_prob: bool = False,
        params: Tensor | None = None,
    ) -> FilteringDistribution | List[FilteringDistribution] | Tuple:
        """
        Run the filter over ``data``.

        Parameters
        ----------
        data : Data
            Observations ``y`` with shape ``(T, ydim)`` or ``(B, T, ydim)`` for a
            batch of independent, equal-length trajectories (optional inputs
            ``u`` with matching leading shape).
        init_dist : FilteringDistribution
            Prior ``p(x_0)``; shared by every trajectory in a batch.
        y0 : bool
            Whether the first data row observes ``x_0`` (see module docstring).
        return_history : bool
            Return the list ``[p(x_k | y_{...k})]_k`` instead of the final
            distribution only.
        compute_log_prob : bool
            Also return ``log p(y_{1:n} | theta)`` (one value per trajectory).
        params : Tensor, optional
            Flat parameter vector forwarded to ``model.update`` first.
        """

        if data.y.ndim > 2 and not self.supports_batch:
            raise NotImplementedError(f"{type(self).__name__} does not support batched trajectories.")

        if params is not None:
            self.model.update(params)

        dist = self.initialize(init_dist)
        history = [] if return_history else None
        log_prob = 0.0

        if return_history and not y0:
            history.append(dist.snapshot())

        for t in range(len(data)):
            if t > 0 or not y0:
                dist = self.predict_step(dist, u=_input_at(data.u, t - 1 if y0 else t))

            dist, log_lik = self.update_step(
                data.y[..., t, :], dist, u=_input_at(data.u, t), compute_log_prob=compute_log_prob,
            )
            if compute_log_prob:
                log_prob = log_prob + log_lik
            if return_history:
                history.append(dist.snapshot())

        self.dist = dist
        output = history if return_history else dist
        return (output, log_prob) if compute_log_prob else output

    def nlog_marginal_likelihood(
        self,
        data: Data,
        init_dist: FilteringDistribution,
        y0: bool = False,
        params: Tensor | None = None
    ) -> Tensor:
        """
        ``-log p(y | theta)`` as a scalar.  For a batch of independent
        trajectories the per-trajectory terms are summed.
        """

        _, log_prob = self.filter(
            data=data,
            init_dist=init_dist,
            y0=y0,
            return_history=False,
            compute_log_prob=True,
            params=params,
        )

        return -torch.as_tensor(log_prob).sum()


class _LinearGaussianObservations:
    """
    Mixin: expose the observation model as a `LinearGaussianModel`, wrapping a
    noiseless `LinearModel` (e.g. `IdentityModel`) on the fly.  The user's
    `SSM` is never modified.
    """

    _obs_source = None
    _obs_wrapped = None

    @property
    def observations(self) -> LinearGaussianModel:
        obs = self.model.observations
        if obs is not self._obs_source:
            self._obs_source = obs
            self._obs_wrapped = _ensure_linear_gaussian(obs, self.model.ydim)
        return self._obs_wrapped


def _require_linear_gaussian_dynamics(model: SSM, name: str) -> None:
    if not isinstance(model.dynamics, LinearGaussianModel):
        raise ValueError(
            f"{name} requires a linear-Gaussian dynamics model, got {type(model.dynamics)}."
        )


# ---------------------------------------------------------------------------
# Concrete Gaussian filters
# ---------------------------------------------------------------------------

class KalmanFilter(_LinearGaussianObservations, Filter):
    """Standard linear-Gaussian Kalman filter."""

    def __init__(
        self,
        model: SSM
    ) -> None:

        _require_linear_gaussian_dynamics(model, "KalmanFilter")
        _ensure_linear_gaussian(model.observations, model.ydim)  # validate early

        super().__init__(
            model=model,
            dynamics_filter=KalmanPredict(),
            observations_filter=KalmanPredict(),
            update=KalmanUpdate(),
        )


class SquareRootKalmanFilter(_LinearGaussianObservations, Filter):
    """
    Square-root Kalman filter.  Same fixed point as the standard KF but never
    materializes a dense covariance: predict and update operate on Cholesky
    factors through QR factorizations, and the marginal-likelihood term is
    read off the innovation factor.  Recommended when the state covariance is
    ill-conditioned.
    """

    def __init__(
        self,
        model: SSM
    ) -> None:

        _require_linear_gaussian_dynamics(model, "SquareRootKalmanFilter")
        _ensure_linear_gaussian(model.observations, model.ydim)

        super().__init__(
            model=model,
            dynamics_filter=SquareRootKalmanPredict(),
        )

    def update_step(self, y, dist, u=None, compute_log_prob=False):
        # Observation predict and update are fused into a single QR.
        return F.srkf_update(y, dist, self.observations, u=u, compute_log_prob=compute_log_prob)


class UnscentedKalmanFilter(Filter):
    """Unscented Kalman filter (Julier & Uhlmann; Wan & van der Merwe weights)."""

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
        self.lmbda = alpha**2 * (model.xdim + kappa) - model.xdim

        super().__init__(
            model=model,
            dynamics_filter=UnscentedKalmanPredict(self.lmbda, regenerate_points=regenerate_points),
            observations_filter=UnscentedKalmanPredict(self.lmbda, regenerate_points=regenerate_points),
            update=GaussQuadKalmanUpdate(),
        )

    def initialize(self, init_dist):
        dist = super().initialize(init_dist)
        dist.form_ut_weights(alpha=self.alpha, beta=self.beta, kappa=self.kappa, lmbda=self.lmbda)
        return dist


class GaussHermiteFilter(Filter):
    """Gauss-Hermite Kalman filter with tensor-product nodes of the given order."""

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

    def initialize(self, init_dist):
        dist = super().initialize(init_dist)
        dist.form_gh_weights(order=self.order)
        return dist


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

    def initialize(self, init_dist):
        dist = super().initialize(init_dist)
        dist.form_cubature_weights()
        return dist


class EnsembleKalmanFilter(Filter):
    """Stochastic ensemble Kalman filter (single trajectory)."""

    supports_batch = False

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

    def initialize(self, init_dist):
        dist = super().initialize(init_dist)
        dist.sample_particles(self.ensemble_size)
        return dist


# ---------------------------------------------------------------------------
# Particle filter (bootstrap with optional ESS-based resampling)
# ---------------------------------------------------------------------------

class ParticleFilter(Filter):
    """
    Bootstrap particle filter with effective-sample-size triggered resampling.

    Predict samples each particle through the dynamics model (``model.sample``),
    so arbitrary non-Gaussian process noise is supported.  Update weights the
    particles by ``log p(y | x_i)``, which needs an additive-Gaussian
    observation model; the log-sum-exp of the weights is the
    marginal-likelihood term (unbiased in ``p(y | theta)``, which is what
    pseudo-marginal MCMC requires).
    """

    supports_batch = False

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

        super().__init__(model=model)
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.ess_threshold = ess_threshold

    def initialize(self, init_dist):
        dist = super().initialize(init_dist)
        dist.sample_particles(self.n_particles)
        return dist

    def predict_step(self, dist, u=None):
        return F.bootstrap_pf_predict(self.dynamics, dist, u=u)

    def update_step(self, y, dist, u=None, compute_log_prob=False):
        return F.particle_filter_update(
            y, dist, self.observations,
            resample_threshold=self.ess_threshold,
            resample_method=self.resample_method,
            u=u,
        )
