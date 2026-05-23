# Welcome to B-Side

A Bayesian system identification (B-SIde) toolbox built on top of PyTorch.

`bside` ships state-space modeling primitives, a family of Gaussian and particle
filters and smoothers, square-root variants for numerical stability, and a small
system-identification module that includes a closed-form linear-Gaussian EM and
an adapter to the [mcmc-samplers](https://github.com/ngalioto/mcmc-samplers)
package.

## Installation

Install the latest release
```bash
pip install bside
```

Or install from source
```bash
pip install git+https://github.com/ngalioto/bside.git
```

Optional extras:
```bash
pip install "bside[dev]"   # pytest + matplotlib
pip install "bside[mcmc]"  # mcmc-samplers (for the Posterior + DRAM workflow)
pip install "bside[docs]"  # mkdocs-material
```

## Quickstart

### Build a state-space model

```python
import torch
import bside

dt, g, qc, r = 0.1, 9.81, 0.1, 0.1
A = bside.Matrix(torch.linalg.matrix_exp(torch.tensor([[0., 1.], [-g, 0.]]) * dt))
C = bside.Matrix(torch.tensor([[1., 0.]]))
Q = bside.PSDMatrix(torch.tensor([[qc*dt**3/3, qc*dt**2/2], [qc*dt**2/2, qc*dt]]))
R = bside.PSDMatrix(torch.tensor([[r**2]]))

dynamics = bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=Q)
observations = bside.LinearGaussianModel(model=bside.LinearModel(C), noise_cov=R)
ssm = bside.SSM(xdim=2, ydim=1, dynamics=dynamics, observations=observations)
```

### Filter

```python
init_dist = bside.FilteringDistribution(mean=torch.tensor([1.5, 0.]),
                                        cov=0.01 * torch.eye(2))
data = bside.Data(y=ssm.measure(x=init_dist.mean, T=200))

filter = bside.KalmanFilter(model=ssm)
history, log_marg = filter.filter(data, init_dist, return_history=True,
                                  compute_log_prob=True)
```

For a *nonlinear* SSM swap in `NonlinearAdditiveModel` and choose any of the
non-linear filters:

| Filter | When to use |
| --- | --- |
| `KalmanFilter` | Linear-Gaussian dynamics + observations |
| `SquareRootKalmanFilter` | Same, but with ill-conditioned covariances |
| `UnscentedKalmanFilter` | Smoothly nonlinear, additive-Gaussian noise |
| `GaussHermiteFilter` | Higher-order quadrature, smooth nonlinearities |
| `CubatureKalmanFilter` | Cheaper 3rd-order alternative to GH |
| `EnsembleKalmanFilter` | High-dimensional state, sampling-based covariances |
| `ParticleFilter` | Strongly nonlinear / non-Gaussian, additive observations |

### Smooth

```python
rts = bside.RTSSmoother(model=ssm)
smoothed = rts.smooth(history)
```

`UnscentedRTSSmoother` works for any additive-Gaussian non-linear dynamics, and
`ParticleSmoother` runs forward-filter / backward-simulation on a `ParticleFilter`
history.

### System identification

```python
# Closed-form EM for linear-Gaussian SSMs
em = bside.EM(ssm=fit_ssm, init_dist=init_dist)
em.fit(data, n_iter=50)

# Bayesian inference: wrap any filter as an mcmc-samplers target
from mcmc_samplers import DelayedRejectionAdaptiveMetropolis
posterior = bside.Posterior(filter=bside.KalmanFilter(model=fit_ssm),
                            data=data, init_dist=init_dist)
dram = DelayedRejectionAdaptiveMetropolis(target=posterior, x0=theta0, cov=cov0)
samples, log_probs = dram(N=10_000)
```

See [`examples/`](https://github.com/ngalioto/bside/tree/main/examples) for full
runnable scripts (Lorenz '63 UKF / PF, linear smoother, EM, Bayesian pendulum).

## License

MIT License (c) 2024

## Cite

If you found this package useful, please consider citing the [paper](https://link.springer.com/article/10.1007/s11071-020-05925-8).

```bibtex
@article{galioto2020bayesian,
  title={Bayesian system {ID}: optimal management of parameter, model, and measurement uncertainty},
  author={Galioto, Nicholas and Gorodetsky, Alex Arkady},
  journal={Nonlinear Dynamics},
  volume={102},
  number={1},
  pages={241--267},
  year={2020},
  publisher={Springer}
}
```
