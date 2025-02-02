"""bside: Bayesian system identification"""

from bside.dataset import DataTrajectories, Data
from bside.dmd import DMD, DMDc
from bside.dynamics import Model, AdditiveModel, LinearModel, NonlinearModel, IdentityModel, NonlinearAdditiveModel, LinearGaussianModel
from bside.filtering import *
from bside.models import FeedforwardNetwork, ResidualNetwork, PSDMatrix, Matrix, SquaredMatrix, ExponentialMatrix, DiagonalMatrix
from bside.ssm import SSM
from bside.subspace_encoder import SubspaceEncoder


__all__ = (
    "AdditiveModel",
    "Data",
    "DataTrajectories",
    "DMD",
    "DMDc",
    "DiagonalMatrix",
    "ExponentialMatrix",
    "FeedforwardNetwork",
    "IdentityModel",
    "LinearGaussianModel",
    "LinearModel",
    "Matrix",
    "Model",
    "NonlinearAdditiveModel",
    "NonlinearModel",
    "PSDMatrix",
    "ResidualNetwork",
    "SSM",
    "SquaredMatrix",
    "SubspaceEncoder"
)