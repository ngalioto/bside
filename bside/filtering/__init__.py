"""
This subpackage provides a collection of Bayesian filtering-related algorithms and helper functions.

Modules:
- `distributions`: Defines the filtering distribution class
- `filters`: Implements various filtering algorithms (Kalman, square-root Kalman,
  Unscented, Gauss-Hermite, Cubature, Ensemble, Particle).
- `functional`: Contains stateless helpers used by the filters and smoothers.
- `smoothers`: RTS / Unscented-RTS / Particle backward passes that consume
  a forward-filtering history.
- `plotting`: Plotting helpers for filtering distributions.
"""

from .distributions import FilteringDistribution

from .filters import (
    CubatureKalmanFilter,
    CubatureKalmanPredict,
    EnsembleKalmanFilter,
    EnsembleKalmanPredict,
    EnsembleKalmanUpdate,
    Filter,
    FilterPredict,
    FilterUpdate,
    GaussHermiteFilter,
    GaussHermitePredict,
    GaussQuadKalmanUpdate,
    KalmanFilter,
    KalmanPredict,
    KalmanUpdate,
    ParticleFilter,
    SquareRootKalmanFilter,
    SquareRootKalmanPredict,
    UnscentedKalmanFilter,
    UnscentedKalmanPredict,
)

from . import functional

from .plotting import collate_filtering_distributions, plot_filtering_distributions

__all__ = [
    'CubatureKalmanFilter',
    'CubatureKalmanPredict',
    'EnsembleKalmanFilter',
    'EnsembleKalmanPredict',
    'EnsembleKalmanUpdate',
    'Filter',
    'FilterPredict',
    'FilterUpdate',
    'FilteringDistribution',
    'GaussHermiteFilter',
    'GaussHermitePredict',
    'GaussQuadKalmanUpdate',
    'KalmanFilter',
    'KalmanPredict',
    'KalmanUpdate',
    'ParticleFilter',
    'SquareRootKalmanFilter',
    'SquareRootKalmanPredict',
    'UnscentedKalmanFilter',
    'UnscentedKalmanPredict',
    'collate_filtering_distributions',
    'functional',
    'plot_filtering_distributions',
]
