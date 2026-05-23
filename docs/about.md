# About

`bside` is a Bayesian system identification (B-SIde) toolbox developed by
Nicholas Galioto and contributors.  The package supports the workflow of
modeling, filtering, smoothing, and parameter inference for state-space
dynamical systems.

## Design goals

* **Composable.**  Filters, smoothers, models, and posteriors compose freely
  with `torch.optim` optimizers and external samplers (e.g. mcmc-samplers).
* **Efficient.**  Covariance representations are lazily cached, Cholesky
  factors are re-used wherever possible, and the square-root filters operate
  directly on Cholesky factors via QR factorizations.
* **Extensible.**  New filters, smoothers, and models slot into the same
  `Filter` / `Smoother` / `Model` abstract bases.

See the [algorithms page](algorithms.md) for a per-algorithm reference, and
the `examples/` directory in the repo for end-to-end runnable scripts.
