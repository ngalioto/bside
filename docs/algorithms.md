# Algorithm reference

This page summarizes the filters, smoothers, and system-identification
algorithms shipped with `bside`, with literature pointers for each.

## Filters

### Kalman filter (`KalmanFilter`)

The textbook linear-Gaussian recursion.  Use this whenever your dynamics and
observations are both linear-Gaussian; it is the only filter that produces an
exact posterior in closed form for the linear case.

References: Kalman, R. E. *A new approach to linear filtering and prediction
problems*, 1960.

### Square-root Kalman filter (`SquareRootKalmanFilter`)

Implements the Park-Kailath square-root form: predict and update operate on
the Cholesky factor of the covariance via QR factorizations of stacked sqrt
matrices.  Strictly equivalent to `KalmanFilter` in exact arithmetic but more
robust when the state covariance becomes ill-conditioned (e.g. very small
process noise, near-singular initial covariance).

References: Park & Kailath, *New square-root algorithms for Kalman filtering*,
IEEE TAC, 1995.

### Unscented Kalman filter (`UnscentedKalmanFilter`)

Nonlinear, additive-Gaussian; propagates a deterministic set of sigma points
through the dynamics and observation models and aggregates the moments.  Set
`regenerate_points=False` to re-use the propagated sigma points in the
observation step (faster, slightly less accurate).

References: Julier & Uhlmann, *Unscented filtering and nonlinear estimation*,
Proc. IEEE, 2004.

### Gauss-Hermite filter (`GaussHermiteFilter`)

Same shape as the UKF but uses tensor-product Gauss-Hermite quadrature for the
sigma points.  More accurate than the UKF for smooth nonlinearities at the
cost of `order ** dim` quadrature points.

References: Ito & Xiong, *Gaussian filters for nonlinear filtering problems*,
IEEE TAC, 2000.

### Cubature Kalman filter (`CubatureKalmanFilter`)

3rd-order spherical-cubature alternative to the UKF: `2n` equally weighted
points, no free tuning parameters.  Typically as accurate as the UKF with the
default parameters and slightly cheaper.

References: Arasaratnam & Haykin, *Cubature Kalman filters*, IEEE TAC, 2009.

### Ensemble Kalman filter (`EnsembleKalmanFilter`)

Monte-Carlo Kalman: propagates an ensemble of state samples and computes
covariances empirically from the ensemble.  Cheap per-step for high-dimensional
states.

References: Evensen, *Sequential data assimilation with a nonlinear
quasi-geostrophic model using Monte Carlo methods*, JGR, 1994.

### Particle filter (`ParticleFilter`)

Bootstrap sequential importance resampling.  Supports systematic, stratified,
and multinomial resampling and ESS-triggered resampling so the particle cloud
doesn't degenerate.  Per-step log-marginal-likelihood is computed via
`logsumexp` over the unnormalized log-weights.

References: Doucet, de Freitas & Gordon, *Sequential Monte Carlo methods in
practice*, 2001; Gordon, Salmond & Smith, 1993.

## Smoothers

### Rauch-Tung-Striebel smoother (`RTSSmoother`)

Linear-Gaussian backward pass that re-uses the ``U = Sigma A^T`` cross-cov
already produced by `kf_predict` to compute the smoothing gain without any
extra matrix multiplies.

References: Rauch, Tung & Striebel, *Maximum likelihood estimates of linear
dynamic systems*, AIAA, 1965.

### Unscented RTS smoother (`UnscentedRTSSmoother`)

Generalizes the RTS recursion to any additive-Gaussian nonlinear dynamics via
the unscented cross-covariance.

References: Sarkka, *Unscented Rauch-Tung-Striebel smoother*, IEEE TAC, 2008.

### Particle smoother (`ParticleSmoother`)

Forward-filter / backward-simulation (FFBS) smoother for additive-Gaussian
dynamics.  Requires a `ParticleFilter` forward pass with `return_history=True`.

References: Doucet, Godsill & Andrieu, *On sequential Monte Carlo sampling
methods for Bayesian filtering*, Stat. Comput., 2000.

## System identification

### Expectation-maximization (`EM`)

Shumway-Stoffer EM for linear-Gaussian SSMs.  E-step uses `KalmanFilter` +
`RTSSmoother` and recovers the lag-one smoothed cross-covariances from the
smoothing gains in the same backward sweep.  M-step has closed-form updates
for ``A``, ``C``, ``Q``, ``R``, and the initial state distribution.

References: Shumway & Stoffer, *An approach to time series smoothing and
forecasting using the EM algorithm*, J. Time Series Anal., 1982.

### Posterior adapter (`Posterior`)

Wraps any `Filter` into a single-arg callable
``params -> log p(y | params) + log p(params)`` that can be plugged directly
into `mcmc_samplers.Sampler` subclasses (e.g. delayed-rejection adaptive
Metropolis).

### Multi-shooting loss (`MultiShootingLoss`)

Decoupled multi-step prediction loss extracted from `SubspaceEncoder.fit` so
it composes with any `torch.optim` optimizer.
