import torch
from torch import Tensor

from typing import Union, Tuple, Type, Callable

class FilteringDistribution:

    def __init__(
        self,
        mean : Tensor,
        cov : Tensor,
        particles : Tensor = None,
        mean_weights : Tensor = None,
        cov_weights : Tensor = None,
        **hyper_params
    ):
        
        self.mean = mean
        self.dim = mean.shape[-1]
        self.cov = cov
        self._particles = particles
        self.size = 0 if particles is None else particles.shape[0]
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.hyper_params = hyper_params

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

    def sample(
        self,
        n : int
    ) -> Tensor:
        
        self.particles = torch.distributions.MultivariateNormal(self.mean, self.cov).sample((n,))
    
    # sigma points for unscented transform
    def ut_points(
        self
    ):
        
        n = self.dim
        try:
            L = torch.linalg.cholesky(self.cov, upper=True)
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

def kalman_update(
    dist : Type[FilteringDistribution],
    U : Tensor,
    v : Tensor,
    sqrtS : Tensor = None,
    S : Tensor = None,
    Sinv : Tensor = None
) -> Type[FilteringDistribution]:
    
    if Sinv is not None:
        K = U @ Sinv
    elif S is not None:
        K = torch.linalg.solve(S, U, left=False)
    elif sqrtS is not None:
        K = torch.linalg.solve_triangular(sqrtS, \
            torch.linalg.solve_triangular(sqrtS, U, upper=False, left=False), \
                upper=False, left=False)
    else:
        raise ValueError('Missing argument. Must pass S, sqrtS, or Sinv')

    dist.mean = dist.mean + torch.squeeze(K @ v)
    dist.cov = dist.cov - torch.bmm(K, U.transpose(-1,-2))
    return dist

def kf_predict(
    model : Callable,
    dist : Type[FilteringDistribution],
    u : Tensor = None,
    crossCov : bool = False,
) -> Union[Tuple[Type[FilteringDistribution], Tensor], Type[FilteringDistribution]]:
    
    dist.mean = model(dist.mean,u)
    U = dist.cov @ model.mat_x.T
    dist.cov = model.mat_x @ U + model.noise_cov

    return (dist,U) if crossCov else dist

def enkf_predict(
    model,
    dist : Type[FilteringDistribution],
    u : Tensor = None,
    crossCov : bool = False
) -> Union[Tuple[Type[FilteringDistribution], Tensor], Type[FilteringDistribution]]:
    
    Y_particles = model.sample(dist.particles,u)
    dist_Y = FilteringDistribution(
        mean = torch.mean(dist.particles, dim=0),
        cov = (dist.particles - dist.mean).T @ (dist.particles - dist.mean) /  \
        (dist.size - 1),
        particles = Y_particles
    )

    U = dist.cov @ model.mat_x.T
    dist.cov = model.mat_x @ U + model.noise_cov

    return (dist_Y, U) if crossCov else dist_Y
    

# Gaussian quadrature
def gaussian_quadrature(
    model,
    dist_X : Type[FilteringDistribution],
    u : Tensor = None,
    crossCov : bool = False,
    additive : bool = False
) -> Union[Tuple[Type[FilteringDistribution], Tensor], Type[FilteringDistribution]]:

    Y = model(dist_X.particles, u)

    Ymean = torch.sum(Y.T * dist_X.mean_weights, 1, keepdims=True).T

    resY = Y - Ymean
    P = (resY.T * dist_X.cov_weights) @ resY
    
    if additive:
        P = P + model.noise_cov

    dist_Y = FilteringDistribution(Ymean, P, Y, dist_X.mean_weights, dist_X.cov_weights)
        
    if crossCov:
        resX = dist_X.particles - \
            torch.sum(dist_X.particles.T * dist_X.mean_weights, 1, keepdims=True).T
        U = (resX.T * dist_X.cov_weights) @ resY
        return dist_Y, U
    
    else:
        return dist_Y
