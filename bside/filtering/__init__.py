"""
This subpackage provides a collection of Bayesian filtering-related algorithms and helper functions.

Modules:
- `distributions`: Defines the filtering distribution class
- `filters`: Implements various filtering algorithms like Kalman and Ensemble Kalman filters.
- `functional`: Contains helper functions for the complete filtering algorithms.
- `plotting`: Contains helper functions for plotting filtering distributions.
"""

from .distributions import FilteringDistribution

from .filters import (
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
    UnscentedKalmanFilter,
    UnscentedKalmanPredict
)

from . import functional

from .plotting import collate_filtering_distributions, plot_filtering_distributions

__all__ = [
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
    'UnscentedKalmanFilter',
    'UnscentedKalmanPredict',
    'collate_filtering_distributions',
    'functional',
    'plot_filtering_distributions',
]