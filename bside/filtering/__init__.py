"""
This subpackage provides a collection of Bayesian filtering-related algorithms and helper functions.

Modules:
- `distributions`: Defines the filtering distribution class
- `filters`: Implements various filtering algorithms like Kalman and Ensemble Kalman filters.
- `functional`: Contains helper functions for the complete filtering algorithms.
"""

from .distributions import FilteringDistribution

from .filters import (
    EnsembleKalmanFilter,
    EnsembleKalmanPredict,
    EnsembleKalmanUpdate,
    Filter,
    FilterPredict,
    FilterUpdate,
    UnscentedKalmanPredict,
    KalmanFilter,
    KalmanPredict,
    KalmanUpdate,
    UnscentedKalmanFilter,
)

from . import functional

__all__ = [
    'EnsembleKalmanFilter',
    'EnsembleKalmanPredict',
    'EnsembleKalmanUpdate',
    'Filter',
    'FilterPredict',
    'FilterUpdate',
    'FilteringDistribution',
    'UnscentedKalmanPredict',
    'KalmanFilter',
    'KalmanPredict',
    'KalmanUpdate',
    'UnscentedKalmanFilter',
    'functional',
]