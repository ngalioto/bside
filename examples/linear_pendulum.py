# %% [markdown]
# # Linear system identification: pendulum
#
# Learn a linear pendulum model from noisy data. The model has the same two
# halves as in the filtering examples (dynamics and observations, each a
# deterministic map plus Gaussian noise), but now the matrices are unknown
# parameters $\theta$.
#
# The filter turns the model into a likelihood: its predict / update recursion
# accumulates $p(y_{1:T} \mid \theta) = \prod_k p(y_k \mid y_{1:k-1}, \theta)$.
# Two ways to use it are shown:
#
# - **EM** finds a point estimate of $(A, \Sigma, C, \Gamma)$: a Kalman filter and RTS smoother
#   (E-step) alternate with closed-form updates (M-step).
# - **Bayesian inference** samples the posterior
#   $p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)$ with delayed-rejection
#   adaptive Metropolis (DRAM). This needs `pip install bside[mcmc]`.

# %%
import torch
import matplotlib.pyplot as plt

import bside
from mcmc_samplers import DelayedRejectionAdaptiveMetropolis, SampleVisualizer

torch.manual_seed(0)

# %% [markdown]
# ## Data from the true system

# %%
dt, g, qc, r = 0.1, 9.81, 0.1, 0.1
T = 50
xdim, ydim = 2, 2

A_true = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt)
Sigma = bside.PSDMatrix(qc * torch.tensor([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]))
Gamma = bside.PSDMatrix(r**2 * torch.eye(ydim))
C = bside.Matrix(torch.eye(ydim))

true_ssm = bside.SSM(
    xdim=xdim, ydim=ydim,
    dynamics=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(A_true)), noise_cov=Sigma),
    observations=bside.LinearGaussianModel(model=bside.LinearModel(C), noise_cov=Gamma),
)

x0 = torch.tensor([1.5, 0.0])
x_true, y = true_ssm.measure(x=x0, T=T, return_x=True)
data = bside.Data(y=y)
init_dist = bside.FilteringDistribution(mean=x0, cov=0.01 * torch.eye(xdim))

t = torch.arange(1, T + 1) * dt
plt.figure(figsize=(8, 3))
plt.plot(t, y, '.', label=['angle', 'angular velocity'])
plt.plot(t, x_true, 'k', lw=1)
plt.xlabel('time')
plt.legend()
plt.show()

# %% [markdown]
# ## Expectation-maximization
#
# Start from $A = I$ with inflated noise covariances. $C$ is also re-estimated,
# so $A$ is identifiable only up to a change of state coordinates. We therefore
# compare its eigenvalues, which do not depend on the coordinates.

# %%
def fresh_ssm(A0, Sigma0, Gamma0):
    return bside.SSM(
        xdim=xdim, ydim=ydim,
        dynamics=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(A0)), noise_cov=bside.PSDMatrix(Sigma0)),
        observations=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(torch.eye(ydim))), noise_cov=bside.PSDMatrix(Gamma0)),
    )


em_ssm = fresh_ssm(torch.eye(xdim), 0.1 * torch.eye(xdim), 0.1 * torch.eye(ydim))
em = bside.EM(ssm=em_ssm, init_dist=init_dist)
log_marg = em.fit(data, n_iter=200, tol=1e-4)

print(f'EM iterations: {len(log_marg)},  log p(y): {log_marg[0]:.2f} -> {log_marg[-1]:.2f}')
print('eigenvalues of A, true:', torch.linalg.eigvals(A_true))
print('eigenvalues of A, EM:  ', torch.linalg.eigvals(em_ssm.dynamics.mat_x))

plt.figure(figsize=(5, 3))
plt.plot(log_marg)
plt.xlabel('EM iteration')
plt.ylabel('log p(y)')
plt.show()

# %% [markdown]
# ## Bayesian inference with DRAM
#
# Here the four entries of $A$ are uncertain and the other matrices are known.
# Marking an entry as learnable takes a `mask`, and `indices` maps each masked
# entry to an element of $\theta$. `Posterior` wraps the Kalman filter's
# marginal likelihood (with a flat prior here) into the target density that
# the sampler evaluates.

# %%
A_learn = bside.Matrix(
    default=torch.eye(xdim),
    mask=torch.ones(xdim, xdim, dtype=torch.bool),
    indices=torch.arange(xdim**2),
)
model = bside.SSM(
    xdim=xdim, ydim=ydim,
    dynamics=bside.LinearGaussianModel(model=bside.LinearModel(A_learn), noise_cov=Sigma),
    observations=true_ssm.observations,
)
posterior = bside.Posterior(filter=bside.KalmanFilter(model), data=data, init_dist=init_dist)

dram = DelayedRejectionAdaptiveMetropolis(
    target=posterior,
    x0=torch.eye(xdim).flatten(),
    cov=1e-3 * torch.eye(xdim**2),
)
with torch.no_grad():
    samples, log_probs = dram(3000)
samples = samples[500:]  # discard burn-in

print(f'acceptance rate: {100 * dram.acceptance_ratio:.1f}%')
print('posterior mean of A:\n', samples.mean(0).reshape(xdim, xdim))
print('posterior std of A:\n', samples.std(0).reshape(xdim, xdim))
print('true A:\n', A_true)

# %%
visualizer = SampleVisualizer(samples)
visualizer.chains()
visualizer.triangular_hist(bins=40)
plt.show()
