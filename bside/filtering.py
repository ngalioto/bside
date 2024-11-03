import torch
from torch import Tensor
from torch.linalg import solve_triangular
from math import log

from bside.models import Matrix
from bside.dynamics import Model, AdditiveModel, LinearModel
from typing import Union, Tuple

"""
Consider making children for Gaussian and particle filters
But maybe the only difference is sample and log_prob?
For particle filter, use mean_weights for both mean and cov.
"""

class FilteringDistribution:

    def __init__(
        self,
        mean : Tensor = None,
        cov : Tensor = None,
        particles : Tensor = None,
        mean_weights : Tensor = None,
        cov_weights : Tensor = None,
        **hyper_params
    ):
        
        if particles is None and (mean is None or cov is None):
            raise ValueError("Both mean and cov must be provided if particles are not provided")
        
        if mean is not None:
            self.dim = mean.shape[-1]
        else:
            self.dim = particles.shape[-1]
        
        self.mean = mean
        self._cov = Matrix(cov) if type(cov) is not Matrix else cov
        self._particles = particles
        self.size = 0 if particles is None else particles.shape[0]
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.hyper_params = hyper_params

    @property
    def cov(
        self
    ):
        return self._cov.val
    
    @cov.setter
    def cov(
        self,
        value : Tensor
    ):
            
        self._cov.val = value

    @property
    def sqrt_cov(
        self
    ):
            
        return self._cov.sqrt
    
    @sqrt_cov.setter
    def sqrt_cov(
        self,
        value : Tensor
    ):
                
        self._cov.sqrt = value

    def update(
        self
    ):
        
        self._cov.update()

    @property
    def particles(
        self
    ):
        
        return self._particles
    
    @particles.setter
    def particles(
        self,
        value : Tensor
    ):
            
        self._particles = value
        self.size = value.shape[0]

    def log_prob(
        self,
        x : Tensor,
        normalize : bool = True
    ) -> Tensor:
        
        """
        TODO: What if the distribution is not Gaussian?

        Compute the Gaussian log probability of a given point x
        """

        v = x - self.mean
        log_prob = torch.sum(solve_triangular(self.sqrt_cov, v.T, upper=False)**2, axis=0)

        if normalize:
            log_prob = log_prob + torch.log(torch.linalg.det(self.cov)) + self.dim * log(2*torch.pi)

        return -0.5 * log_prob
    

    def sample(
        self,
        n : int
    ) -> Tensor:
        
        self.particles = torch.randn(n, self.dim) @ self.sqrt_cov.T + self.mean
    
    # sigma points for unscented transform
    def ut_points(
        self
    ):
        
        n = self.dim
        try:
            L = self.sqrt_cov   # this might need to be upper triangular
        except torch.linalg.LinAlgError: #P not positive-definite
            xout = None
        else:
            scaling = torch.sqrt(n + self.hyper_params["lmbda"])
            scaledL = L*scaling
            xout = torch.zeros(2 * n + 1, n)
            xout[0] = self.mean
            xout[1:n+1] = self.mean + scaledL
            xout[n+1:] = self.mean - scaledL
        self.particles = xout

    # weights for unscented transform
    def ut_weights(
        self,
        n : int,
        alpha : float,
        beta : float,
        kappa : float
    ):
        
        lmbda = torch.tensor([alpha**2*(n+kappa) - n])
        Wm = torch.zeros(2*n+1)
        Wc = torch.zeros(2*n+1)

        Wm[0] = lmbda / (n+lmbda)
        Wm[1:] = 1 / (2*(n+lmbda))
        Wc[0] = lmbda / (n+lmbda) + 1 - alpha**2 + beta
        Wc[1:] = 1 / (2*(n+lmbda))
        self.mean_weights = Wm
        self.cov_weights = Wc
        self.hyper_params["lmbda"] = lmbda

def kalman_gain(
    dist : FilteringDistribution,
    U : Tensor,
    Sinv : Tensor = None
) -> Tensor:

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
    y : Tensor,
    dist_x : FilteringDistribution,
    dist_y : FilteringDistribution,
    U : Tensor,
    Sinv : Tensor = None
) -> FilteringDistribution:
    
    K = kalman_gain(dist_y, U, Sinv)

    dist_x.mean = dist_x.mean + K @ (y - dist_y.mean)    #torch.squeeze(K @ v)
    dist_x.cov = dist_x.cov - K @ U.T                    #torch.bmm(K, U.transpose(-1,-2))
    return dist_x

def kf_predict(
    model : LinearModel,
    dist : FilteringDistribution,
    u : Tensor = None,
    crossCov : bool = False,
) -> Union[Tuple[FilteringDistribution, Tensor], FilteringDistribution]:
    
    U = dist.cov @ model.mat_x.T
    dist = FilteringDistribution(
        mean = model(dist.mean, u),
        cov = model.mat_x @ U + model.noise_cov
    )

    return (dist,U) if crossCov else dist

def enkf_predict(
    model : Model,
    dist : FilteringDistribution,
    u : Tensor = None,
    crossCov : bool = False
) -> Union[Tuple[FilteringDistribution, Tensor], FilteringDistribution]:
    
    """
    EnKF predict

    Parameters
    ----------
    model : Model
        The model to use for prediction
    dist : FilteringDistribution
        The distribution to predict
    u : Tensor, optional
        The control input, by default None
    crossCov : bool, optional
        Whether to return the cross covariance, by default False

    Returns
    -------
    Union[Tuple[FilteringDistribution, Tensor], FilteringDistribution]
        The predicted distribution and the cross covariance `U` if crossCov is True

    """
    
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
    y : Tensor,
    dist_x : FilteringDistribution,
    dist_y : FilteringDistribution,
    U : Tensor,
    Sinv : Tensor = None
) -> FilteringDistribution:
        
    v = y - dist_y.particles
    K = kalman_gain(dist_y, U, Sinv)

    dist_x.particles = dist_x.particles + torch.einsum('ij,bj->bi', K, v)

    # Do not necessarily need the mean and cov for filtering
    dist_x.mean = None
    dist_x.cov = None

    return dist_x
    

# Gaussian quadrature
def gaussian_quadrature(
    model : Model,
    dist_X : FilteringDistribution,
    u : Tensor = None,
    crossCov : bool = False,
) -> Union[Tuple[FilteringDistribution, Tensor], FilteringDistribution]:
    
    additive = type(model) is AdditiveModel

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
        res_X = dist_X.particles - \
            torch.sum(dist_X.particles.T * dist_X.mean_weights, 1, keepdims=True).T
        U = (res_X.T * dist_X.cov_weights) @ res_Y
        return dist_Y, U
    
    else:
        return dist_Y
