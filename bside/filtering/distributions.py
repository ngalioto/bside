import torch
from torch import Tensor
from torch.linalg import solve_triangular
from numpy.polynomial.hermite import hermgauss
from math import log, sqrt, pi

from bside.models import PSDMatrix


class FilteringDistribution:

    """
    Container for a (possibly weighted) particle / sigma-point representation of
    a filtering distribution.  Stores any subset of:

    * ``mean`` and ``cov`` (Gaussian sufficient statistics)
    * ``particles`` -- the locations of the support
    * ``log_weights`` -- weighted-particle (importance) weights for the support
    * ``quad_points`` and ``mean_weights`` / ``cov_weights`` -- deterministic
      quadrature nodes plus their (typically unequal) weights for the mean and
      covariance

    Particle filters operate on the (particles, log_weights) representation;
    Gaussian filters operate on (mean, cov); deterministic-quadrature filters
    (UT, Gauss-Hermite, cubature) use both at once.
    """

    def __init__(
        self,
        mean: Tensor = None,
        cov: Tensor | PSDMatrix = None,
        particles: Tensor = None,
        quad_points: Tensor = None,
        mean_weights: Tensor = None,
        cov_weights: Tensor = None,
        log_weights: Tensor = None,
        sqrt_cov: Tensor = None,
        **hyper_params
    ) -> None:
        
        # Accept either dense cov, square-root cov, or pure-particle representation.
        if particles is None and mean is None:
            raise ValueError("Provide either particles or (mean, cov/sqrt_cov).")
        if particles is None and cov is None and sqrt_cov is None:
            raise ValueError("Particle-free distributions must provide cov or sqrt_cov.")
        
        if mean is not None:
            self.dim = mean.shape[-1]
        else:
            self.dim = particles.shape[-1]
        
        self.mean = mean
        if cov is None and sqrt_cov is None:
            self._cov = None
        elif isinstance(cov, PSDMatrix):
            self._cov = cov
        elif cov is not None:
            self._cov = PSDMatrix(cov)
        else:
            # square-root only construction
            self._cov = PSDMatrix(default_sqrt=sqrt_cov)
        self._particles = particles
        self.quad_points = quad_points
        self.size = 0 if particles is None else particles.shape[0]
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.log_weights = log_weights
        self.hyper_params = hyper_params

    @property
    def cov(
        self
    ) -> Tensor | None:
        return None if self._cov is None else self._cov.val
    
    @cov.setter
    def cov(
        self,
        value: Tensor | None
    ) -> None:

        if value is None:
            self._cov = None
        elif self._cov is None:
            self._cov = value if isinstance(value, PSDMatrix) else PSDMatrix(value)
        elif isinstance(value, PSDMatrix):
            self._cov = value
        else:
            self._cov.val = value
    
    @property
    def sqrt_cov(
        self
    ) -> Tensor | None:
            
        return None if self._cov is None else self._cov.sqrt
    
    @sqrt_cov.setter
    def sqrt_cov(
        self,
        value: Tensor
    ) -> None:
        
        if self._cov is None:
            self._cov = PSDMatrix(default_sqrt=value)
        else:
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
        
        if self._cov is not None:
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
        self.size = value.shape[0] if value is not None else 0

    # ------------------------------------------------------------------
    # Gaussian likelihood
    # ------------------------------------------------------------------

    def log_prob(
        self,
        x: Tensor,
        normalize: bool = True
    ) -> Tensor:
        
        """
        Gaussian log probability at ``x``.  Uses ``solve_triangular`` on the
        Cholesky factor for the Mahalanobis term so we never form the inverse.
        """

        v = torch.atleast_2d(x - self.mean)
        log_prob = torch.sum(solve_triangular(self.sqrt_cov, v.T, upper=False)**2, axis=-2) # Mahalanobis distance

        if normalize:
            log_det = 2 * torch.sum(torch.log(torch.diagonal(self.sqrt_cov, dim1=-2, dim2=-1)), axis=-1)
            log_prob = log_prob + log_det + self.dim * log(2*pi)

        return -0.5 * log_prob
    
    def sample(
        self,
        n: int
    ) -> Tensor:
        
        """Draw ``n`` samples from this Gaussian via the cached Cholesky factor."""
        
        return torch.randn(n, self.dim) @ self.sqrt_cov.T + self.mean

    def sample_particles(
        self,
        n: int
    ) -> None:
        
        """Resample the support of this distribution from N(mean, cov)."""

        self.particles = self.sample(n)
        # Equal-weight particles after fresh sampling.
        self.log_weights = -log(n) * torch.ones(n)

    # ------------------------------------------------------------------
    # Weighted-particle utilities (used by ParticleFilter)
    # ------------------------------------------------------------------

    def normalized_weights(
        self
    ) -> Tensor:
        """Numerically stable normalized weights from ``log_weights``."""

        if self.log_weights is None:
            raise ValueError("log_weights have not been set on this distribution.")
        return torch.softmax(self.log_weights, dim=0)

    def effective_sample_size(
        self
    ) -> Tensor:
        """Kong, Liu, Wong (1994) ESS = 1 / sum(w^2)."""

        w = self.normalized_weights()
        return 1.0 / torch.sum(w * w)

    def weighted_mean(
        self
    ) -> Tensor:
        """Weighted-particle estimate of the mean (without overwriting ``self.mean``)."""

        w = self.normalized_weights()
        return torch.einsum('i,ij->j', w, self.particles)

    def weighted_cov(
        self,
        mean: Tensor | None = None
    ) -> Tensor:
        """Weighted-particle estimate of the covariance."""

        w = self.normalized_weights()
        if mean is None:
            mean = torch.einsum('i,ij->j', w, self.particles)
        res = self.particles - mean
        return (res.T * w) @ res

    def resample(
        self,
        method: str = 'systematic'
    ) -> None:
        """
        Resample the particles in-place, reducing to equal weights.  Supported
        methods are ``'systematic'`` (default, O(N), low-variance), ``'stratified'``,
        and ``'multinomial'``.
        """

        if self.particles is None or self.log_weights is None:
            raise ValueError("resample() requires both particles and log_weights to be set.")

        n = self.size
        w = self.normalized_weights()

        if method == 'multinomial':
            idx = torch.multinomial(w, n, replacement=True)
        elif method == 'stratified':
            u = (torch.arange(n) + torch.rand(n)) / n
            cumw = torch.cumsum(w, dim=0)
            idx = torch.searchsorted(cumw, u).clamp(max=n - 1)
        elif method == 'systematic':
            # Single uniform draw, common offset for every stratum.
            u = (torch.arange(n) + torch.rand(1)) / n
            cumw = torch.cumsum(w, dim=0)
            idx = torch.searchsorted(cumw, u).clamp(max=n - 1)
        else:
            raise ValueError(
                f"Unknown resampling method '{method}'. Use 'systematic', "
                "'stratified', or 'multinomial'."
            )

        self.particles = self.particles[idx].clone()
        self.log_weights = -log(n) * torch.ones(n)

    # ------------------------------------------------------------------
    # Quadrature point generators
    # ------------------------------------------------------------------

    def form_ut_points(
        self,
        lmbda: float
    ) -> None:
        """Unscented transform sigma points around (mean, cov)."""
        
        n = self.dim
        L, info = torch.linalg.cholesky_ex(self.cov)
        if info.any():
            self.particles = None
            return
        scaling = sqrt(n + lmbda)
        scaledL = L * scaling
        self.particles = torch.zeros(2 * n + 1, n)
        self.particles[0] = self.mean
        self.particles[1:n+1] = self.mean + scaledL
        self.particles[n+1:] = self.mean - scaledL

    def form_ut_weights(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        lmbda: float | None = None
    ) -> None:
        """Mean/covariance weights for the unscented transform.

        When ``alpha=1``, ``beta=2``, ``kappa=0`` this reduces to spherical
        Gaussian quadrature.
        """
        
        lmbda = alpha**2 * (self.dim + kappa) - self.dim if lmbda is None else lmbda
        Wm = torch.zeros(2 * self.dim + 1)
        Wc = torch.zeros(2 * self.dim + 1)

        Wm[0] = lmbda / (self.dim + lmbda)
        Wm[1:] = 1 / (2 * (self.dim + lmbda))
        Wc[0] = lmbda / (self.dim + lmbda) + 1 - alpha**2 + beta
        Wc[1:] = 1 / (2 * (self.dim + lmbda))
        self.mean_weights = Wm
        self.cov_weights = Wc

    def form_gh_points(
        self
    ) -> None:
        """Gauss-Hermite quadrature nodes scaled to (mean, cov)."""
        
        L, info = torch.linalg.cholesky_ex(self.cov)
        if info.any():
            self.particles = None
            return
        scaledL = sqrt(2) * L
        self.particles = self.quad_points @ scaledL.T + self.mean

    def form_gh_weights(
        self,
        order: int = 3
    ) -> None:
        """Tensor-product Gauss-Hermite weights of the requested order.

        Uses ``numpy.polynomial.hermite.hermgauss`` for the 1-D nodes/weights
        (cheap, accurate) and a single ``meshgrid + prod`` for the tensor product.
        The grid scales as ``order ** dim`` -- consider sparse-grid (Smolyak)
        rules for ``dim`` larger than ~5.
        """
        
        nodes, weights = hermgauss(deg=order)
        nodes = torch.from_numpy(nodes).float()
        weights = torch.from_numpy(weights).float() / sqrt(pi)

        mesh = torch.meshgrid([nodes] * self.dim, indexing='ij')
        points = torch.stack(mesh, dim=-1).reshape(-1, self.dim)

        weight_mesh = torch.meshgrid([weights] * self.dim, indexing='ij')
        weight_grid = torch.prod(torch.stack(weight_mesh, dim=-1), dim=-1).reshape(-1)

        self.quad_points = points
        self.mean_weights = weight_grid
        self.cov_weights = weight_grid

    def form_cubature_points(
        self
    ) -> None:
        """3rd-order spherical cubature points around (mean, cov).

        Arasaratnam & Haykin (2009), 2n points at +/- sqrt(n) * e_i.
        """

        n = self.dim
        L, info = torch.linalg.cholesky_ex(self.cov)
        if info.any():
            self.particles = None
            return
        scaledL = sqrt(n) * L
        self.particles = torch.zeros(2 * n, n)
        self.particles[:n] = self.mean + scaledL
        self.particles[n:] = self.mean - scaledL

    def form_cubature_weights(
        self
    ) -> None:
        """Equal weights ``1/(2n)`` on each of the 2n cubature points."""

        w = torch.ones(2 * self.dim) / (2 * self.dim)
        self.mean_weights = w
        self.cov_weights = w
