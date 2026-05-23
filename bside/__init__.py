"""bside: Bayesian system identification"""

from bside.dataset import DataTrajectories, Data
from bside.dmd import DMD, DMDc
from bside.dynamics import (
    AdditiveModel,
    IdentityModel,
    LinearGaussianModel,
    LinearModel,
    Model,
    NonlinearAdditiveModel,
    NonlinearModel,
)
from bside.filtering import *  # noqa: F401, F403
from bside.models import (
    DiagonalMatrix,
    ExponentialMatrix,
    FeedforwardNetwork,
    Matrix,
    PSDMatrix,
    ResidualNetwork,
    SquaredMatrix,
)
from bside.ssm import SSM
from bside.subspace_encoder import SubspaceEncoder


__all__ = (
    "AdditiveModel",
    "CubatureKalmanFilter",
    "Data",
    "DataTrajectories",
    "DMD",
    "DMDc",
    "DiagonalMatrix",
    "EnsembleKalmanFilter",
    "ExponentialMatrix",
    "FeedforwardNetwork",
    "FilteringDistribution",
    "GaussHermiteFilter",
    "IdentityModel",
    "KalmanFilter",
    "LinearGaussianModel",
    "LinearModel",
    "Matrix",
    "Model",
    "NonlinearAdditiveModel",
    "NonlinearModel",
    "ParticleFilter",
    "PSDMatrix",
    "ResidualNetwork",
    "SSM",
    "SquareRootKalmanFilter",
    "SquaredMatrix",
    "SubspaceEncoder",
    "UnscentedKalmanFilter",
    "collate_filtering_distributions",
    "plot_filtering_distributions",
)
