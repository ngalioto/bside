import copy

import torch
from torch import Tensor
from torch.linalg import solve_triangular
from numpy.polynomial.hermite import hermgauss
from math import log, sqrt, pi

from bside.models import PSDMatrix


class Covariance:
    """
    Covariance of a state distribution with lazily cached dense, Cholesky and
    inverse forms.  Same interface as `PSDMatrix`, but a plain object rather
    than an ``nn.Module``: filters create one per step, and model covariances
    (which carry learnable parameters) are the only ones that need to be
    modules.
    """

    def __init__(
        self,
        val: Tensor | None = None,
        sqrt: Tensor | None = None
    ) -> None:

        self._val, self._sqrt_val, self._inv_val = val, sqrt, None
        self._up_to_date = val is not None
        self._sqrt_up_to_date = sqrt is not None
        self._inv_up_to_date = False

    @property
    def val(self) -> Tensor:
        if not self._up_to_date:
            self._val = self._sqrt_val @ self._sqrt_val.mT
            self._up_to_date = True
        return self._val

    @property
    def sqrt(self) -> Tensor:
        if not self._sqrt_up_to_date:
            self._sqrt_val = torch.linalg.cholesky(self._val, upper=False)
            self._sqrt_up_to_date = True
        return self._sqrt_val

    @property
    def inv(self) -> Tensor:
        if not self._inv_up_to_date:
            self._inv_val = torch.cholesky_inverse(self.sqrt, upper=False)
            self._inv_up_to_date = True
        return self._inv_val

    def update(self) -> None:
        """No learnable parameters; present for interface parity with `PSDMatrix`."""


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
            self._cov = Covariance(val=cov)
        else:
            # square-root only construction
            self._cov = Covariance(sqrt=sqrt_cov)
        self._particles = particles
        self.quad_points = quad_points
        self.size = 0 if particles is None else particles.shape[-2]
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

        if value is None or isinstance(value, PSDMatrix):
            self._cov = value
        else:
            self._cov = Covariance(val=value)
    
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
        
        self._cov = Covariance(sqrt=value)
    
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

    def snapshot(
        self
    ) -> "FilteringDistribution":
        """
        Cheap, independent copy for storing a filtering history.

        Tensors are shared (filters never modify them in place), but the
        attribute dictionary and the covariance cache are copied, so later
        reassignments on ``self`` do not leak into the snapshot.  This replaces
        a per-timestep ``copy.deepcopy``.
        """

        new = object.__new__(FilteringDistribution)
        new.__dict__ = self.__dict__.copy()
        if self._cov is not None:
            new._cov = copy.copy(self._cov)
        new.hyper_params = dict(self.hyper_params)
        return new

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
        self.size = value.shape[-2] if value is not None else 0

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

        Broadcasts over leading (batch) dimensions of ``x``, ``mean`` and
        ``cov``: a single point gives a 0-D tensor, a ``(..., dim)`` input gives
        a ``(...)`` tensor of log-densities.
        """

        L = self.sqrt_cov
        v = x - self.mean
        z = solve_triangular(L, v.unsqueeze(-1), upper=False).squeeze(-1)
        log_prob = torch.sum(z * z, dim=-1)  # Mahalanobis distance

        if normalize:
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
            log_prob = log_prob + log_det + self.dim * log(2 * pi)

        return -0.5 * log_prob

    def sample(
        self,
        n: int
    ) -> Tensor:
        
        """Draw ``n`` samples from this Gaussian via the cached Cholesky factor."""
        
        return torch.randn(n, self.dim) @ self.sqrt_cov.mT + self.mean

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

    def _scaled_sqrt_cov(
        self,
        scale: float
    ) -> Tensor:
        """
        ``scale * L^T`` where ``L`` is the (cached) lower Cholesky factor of the
        covariance.  Row ``i`` of the result is ``scale`` times the ``i``-th
        *column* of ``L``, which is the direction used by sigma / cubature
        points:  x_i = m + scale * L[:, i].
        """

        try:
            L = self.sqrt_cov
        except torch.linalg.LinAlgError as err:
            raise ValueError(
                "Cannot form quadrature points: the covariance is not positive definite."
            ) from err
        return scale * L.mT

    def form_ut_points(
        self,
        lmbda: float
    ) -> None:
        """Unscented transform sigma points around (mean, cov): m, m +/- sqrt(n+lambda) L[:, i]."""

        offsets = self._scaled_sqrt_cov(sqrt(self.dim + lmbda))  # (..., n, n)
        m = self.mean.unsqueeze(-2)                               # (..., 1, n)
        self.particles = torch.cat([m, m + offsets, m - offsets], dim=-2)

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
        """Gauss-Hermite quadrature nodes scaled to (mean, cov): x = m + sqrt(2) L xi."""

        self.particles = self.quad_points @ self._scaled_sqrt_cov(sqrt(2)) + self.mean.unsqueeze(-2)

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

        Arasaratnam & Haykin (2009), 2n points m +/- sqrt(n) L[:, i].
        """

        offsets = self._scaled_sqrt_cov(sqrt(self.dim))
        m = self.mean.unsqueeze(-2)
        self.particles = torch.cat([m + offsets, m - offsets], dim=-2)

    def form_cubature_weights(
        self
    ) -> None:
        """Equal weights ``1/(2n)`` on each of the 2n cubature points."""

        w = torch.ones(2 * self.dim) / (2 * self.dim)
        self.mean_weights = w
        self.cov_weights = w
