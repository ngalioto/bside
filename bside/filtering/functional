from typing import Tuple
import torch
from torch import Tensor
from torch.linalg import solve_triangular

from bside.filtering.distributions import FilteringDistribution
from bside.dynamics import Model, AdditiveModel, LinearGaussianModel


def kalman_gain(
    dist: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor | None = None
) -> Tensor:
    
    """
    TODO: Test if torch.linalg.solve is actually faster than taking cholesky and then solve_triangular
    """

    if Sinv is not None:
        K = U @ Sinv
    elif not dist._cov._sqrt_up_to_date:
        # might consider using scipy solve with assume_a='sym'
        K = torch.linalg.solve(dist.cov, U, left=False)
    else:
        K = solve_triangular(
            dist.sqrt_cov, \
            solve_triangular(dist.sqrt_cov, U, upper=False, left=False), \
                upper=False, left=False \
            )

    return K

def kalman_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    dist_y: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor | None = None
) -> FilteringDistribution:
    
    K = kalman_gain(dist_y, U, Sinv)

    dist_x.mean = dist_x.mean + K @ (y - dist_y.mean)    #torch.squeeze(K @ v)
    dist_x.cov = dist_x.cov - K @ U.T                    #torch.bmm(K, U.transpose(-1,-2))
    return dist_x

def kf_predict(
    model: LinearGaussianModel,
    dist: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False,
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
    
    U = dist.cov @ model.mat_x.T
    dist = FilteringDistribution(
        mean = model(dist.mean, u),
        cov = model.mat_x @ U + model.noise_cov
    )

    return (dist,U) if crossCov else dist

def enkf_predict(
    model: Model,
    dist: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
    
    """
    EnKF predict

    Parameters
    ----------
    model: Model
        The model to use for prediction
    dist: FilteringDistribution
        The distribution to predict
    u: Tensor, optional
        The control input, by default None
    crossCov: bool, optional
        Whether to return the cross covariance, by default False

    Returns
    -------
    Tuple[FilteringDistribution, Tensor] | FilteringDistribution
        The predicted distribution and the cross covariance `U` if crossCov is True

    """
    
    if isinstance(model, LinearGaussianModel):
        # TODO: Add this in properly and allow for the computation of just square root for efficiency
        # This will reduce variance of estimate
        # should just sample noise values ig
        # dist_Y = FilteringDistribution(
        #     mean = model(dist.mean, u),
        #     sqrt_noise_cov = model.mat_x @ dist.sqrt_noise_cov
        # )
        # dist_Y.sqrt_cov = None
        pass
    else:
        dist_Y = FilteringDistribution(particles=model.sample(dist.particles, u))

    if crossCov:
        dist_Y.mean = torch.mean(dist_Y.particles, 0)
        res_Y = dist_Y.particles - dist_Y.mean
        dist_Y.cov = (res_Y.T @ res_Y) / (dist_Y.size - 1)

        if dist.mean is None:
            dist.mean = torch.mean(dist.particles, 0)
        U = ((dist.particles - dist.mean).T @ res_Y) / (dist.size - 1)

    return (dist_Y, U) if crossCov else dist_Y

def enkf_update(
    y: Tensor,
    dist_x: FilteringDistribution,
    dist_y: FilteringDistribution,
    U: Tensor,
    Sinv: Tensor = None
) -> FilteringDistribution:
        
    v = y - dist_y.particles
    K = kalman_gain(dist_y, U, Sinv)

    dist_x.particles = dist_x.particles + torch.einsum('ij, bj -> bi', K, v)

    # Do not necessarily need the mean and cov for filtering
    dist_x.mean = None
    dist_x.cov = None

    return dist_x
    

# Gaussian quadrature
def gaussian_quadrature(
    model: Model,
    dist_X: FilteringDistribution,
    u: Tensor = None,
    crossCov: bool = False,
) -> Tuple[FilteringDistribution, Tensor] | FilteringDistribution:
    
    additive = isinstance(model, AdditiveModel)

    Y = model(dist_X.particles, u) if additive else model.sample(dist_X.particles, u)

    Ymean = torch.sum(Y.T * dist_X.mean_weights, 1, keepdims=True).T

    res_Y = Y - Ymean
    P = (res_Y.T * dist_X.cov_weights) @ res_Y
    
    if additive:
        P = P + model.noise_cov

    dist_Y = FilteringDistribution(
        mean=Ymean, 
        cov=P, 
        particles=Y,
        mean_weights=dist_X.mean_weights,
        cov_weights=dist_X.cov_weights
    )
        
    if crossCov:
        res_X = dist_X.particles - torch.sum(dist_X.particles.T * dist_X.mean_weights, 1, keepdims=True).T
        U = (res_X.T * dist_X.cov_weights) @ res_Y
        return dist_Y, U
    
    else:
        return dist_Y