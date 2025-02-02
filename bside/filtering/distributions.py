import torch
from torch import Tensor
from torch.linalg import solve_triangular
from math import log, sqrt

from bside.models import PSDMatrix

"""
Consider making children for Gaussian and particle filters
But maybe the only difference is sample and log_prob?
For particle filter, use mean_weights for both mean and cov.
"""

class FilteringDistribution:

    def __init__(
        self,
        mean: Tensor = None,
        cov: Tensor = None,
        particles: Tensor = None,
        mean_weights: Tensor = None,
        cov_weights: Tensor = None,
        **hyper_params
    ) -> None:
        
        if particles is None and (mean is None or cov is None):
            raise ValueError("Both mean and cov must be provided if particles are not provided")
        
        if mean is not None:
            self.dim = mean.shape[-1]
        else:
            self.dim = particles.shape[-1]
        
        self.mean = mean
        self._cov = PSDMatrix(cov) if type(cov) is not PSDMatrix else cov
        self._particles = particles
        self.size = 0 if particles is None else particles.shape[0]
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.hyper_params = hyper_params

    @property
    def cov(
        self
    ) -> Tensor:
        return self._cov.val
    
    @cov.setter
    def cov(
        self,
        value: Tensor
    ) -> None:
            
        self._cov.val = value

    @property
    def sqrt_cov(
        self
    ) -> Tensor:
            
        return self._cov.sqrt
    
    @sqrt_cov.setter
    def sqrt_cov(
        self,
        value: Tensor
    ) -> None:
                
        self._cov.sqrt = value

    @property
    def inv_cov(
        self
    ) -> Tensor:
            
        return self._cov.inv
    
    @inv_cov.setter
    def inv_cov(
        self,
        value: Tensor
    ) -> None:
                
        raise ValueError("Cannot set the covariance matrix inverse directly")

    def update(
        self
    ) -> None:
        
        self._cov.update()

    @property
    def particles(
        self
    ) -> Tensor:
        
        return self._particles
    
    @particles.setter
    def particles(
        self,
        value: Tensor | None
    ) -> None:
            
        self._particles = value
        self.size =value.shape[0] if value is not None else 0

    def log_prob(
        self,
        x: Tensor,
        normalize: bool = True
    ) -> Tensor:
        
        """
        TODO: What if the distribution is not Gaussian?

        Compute the Gaussian log probability of a given point x
        """

        v = torch.atleast_2d(x - self.mean)
        log_prob = torch.sum(solve_triangular(self.sqrt_cov, v.T, upper=False)**2, axis=-2) # Mahalanobis distance

        if normalize:
            log_det = 2 * torch.sum(torch.log(torch.diagonal(self.sqrt_cov, dim1=-2, dim2=-1)), axis=-1) # log determinant
            log_prob = log_prob + log_det + self.dim * log(2*torch.pi)

        return -0.5 * log_prob
    
    def sample(
        self,
        n: int
    ) -> Tensor:
        
        """
        TODO: What if the distribution is not Gaussian?
        """
        
        return torch.randn(n, self.dim) @ self.sqrt_cov.T + self.mean

    def sample_particles(
        self,
        n: int
    ) -> Tensor:
        
        self.particles = self.sample(n)


    # sigma points for unscented transform
    def form_ut_points(
        self,
        lmbda: float
    ) -> None:
        
        n = self.dim
        try:
            L = self.sqrt_cov
        except torch.linalg.LinAlgError: #P not positive-definite
            xout = None
        else:
            scaling = sqrt(n + lmbda)
            scaledL = L * scaling
            xout = torch.zeros(2 * n + 1, n)
            xout[0] = self.mean
            xout[1:n+1] = self.mean + scaledL
            xout[n+1:] = self.mean - scaledL
        self.particles = xout

    # weights for unscented transform
    def form_ut_weights(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        lmbda: float | None = None
    ) -> None:
        
        lmbda = alpha**2 * (self.dim + kappa) - self.dim if lmbda is None else lmbda
        Wm = torch.zeros(2 * self.dim + 1)
        Wc = torch.zeros(2 * self.dim + 1)

        Wm[0] = lmbda / (self.dim + lmbda)
        Wm[1:] = 1 / (2 * (self.dim + lmbda))
        Wc[0] = lmbda / (self.dim + lmbda) + 1 - alpha**2 + beta
        Wc[1:] = 1 / (2 * (self.dim + lmbda))
        self.mean_weights = Wm
        self.cov_weights = Wc