# %% [markdown]
# # Linear system identification: pendulum
#
# Learn a linear pendulum model from sparse, noisy data, following the
# high-noise case of Galioto & Gorodetsky (2020), Sec. 6.3: 40 measurements of
# both states over 4 s with noise standard deviation 0.1. The model has the same
# two halves as in the filtering examples, but now everything is unknown:
#
# - **dynamics**: $x_k = A x_{k-1} + \xi_k,\quad \xi_k \sim \mathcal{N}(0, \sigma_\xi^2 I)$
# - **observations**: $y_k = x_k + \eta_k,\quad \eta_k \sim \mathcal{N}(0, \sigma_\eta^2 I)$
#
# with parameters $\theta = (A_{11}, A_{12}, A_{21}, A_{22}, \log\sigma_\xi, \log\sigma_\eta)$.
# The Kalman filter turns the model into the likelihood
# $p(y_{1:n} \mid \theta) = \prod_k p(y_k \mid y_{1:k-1}, \theta)$. The example
# uses it in four ways: EM, MAP estimation, and posterior sampling with DRAM
# and with HMC. DRAM needs `pip install bside[mcmc]`.

# %%
import math

import torch
import matplotlib.pyplot as plt

import bside
from mcmc_samplers import DelayedRejectionAdaptiveMetropolis, SampleVisualizer

torch.manual_seed(0)

# %% [markdown]
# ## Data
#
# The true system is the exactly discretized linear pendulum (no process noise).

# %%
dt, g, sigma = 0.1, 9.81, 0.1
n = 40
xdim = ydim = 2

A_true = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt)
x0 = torch.tensor([0.1, -0.5])
x_true = bside.LinearModel(bside.Matrix(A_true)).predict(x0, T=n)
y = x_true + sigma * torch.randn_like(x_true)
data = bside.Data(y=y)
init_dist = bside.FilteringDistribution(mean=x0, cov=0.01 * torch.eye(xdim))

t = torch.arange(1, n + 1) * dt
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for i, (ax, name) in enumerate(zip(axes, ['angle', 'angular velocity'])):
    ax.plot(t, x_true[:, i], 'k', lw=1, label='truth')
    ax.plot(t, y[:, i], '.', color='C0', label='data')
    ax.set_title(name)
    ax.set_xlabel('time')
axes[0].legend()
plt.show()

# %% [markdown]
# ## Model with learnable parameters
#
# Each learnable matrix has a `mask` of learnable entries and `indices` that say
# which element of $\theta$ fills each entry. A `CholeskyMatrix` parameterizes a
# covariance by its log-Cholesky factor. With a diagonal mask and a shared
# index, entry $\theta_j$ is $\log\sigma$ and the covariance is $\sigma^2 I$.
# The priors are flat on $A$ and half-normal on the two variances.

# %%
diag = torch.eye(2, dtype=torch.bool)
A = bside.Matrix(torch.eye(2), mask=torch.ones(2, 2, dtype=torch.bool), indices=torch.arange(4))
Sigma = bside.CholeskyMatrix(0.01 * torch.eye(2), mask=diag, indices=torch.tensor([4, 4]))
Gamma = bside.CholeskyMatrix(0.01 * torch.eye(2), mask=diag, indices=torch.tensor([5, 5]))
model = bside.SSM(
    xdim=xdim, ydim=ydim,
    dynamics=bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=Sigma),
    observations=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(torch.eye(2))), noise_cov=Gamma),
)


def log_prior(theta, tau=1.0):
    """Half-normal(tau) on each variance, written for the log-std parameters (incl. Jacobian)."""
    v = torch.exp(2 * theta[..., 4:])
    return (-0.5 * (v / tau) ** 2 + torch.log(2 * v)).sum(-1)


posterior = bside.Posterior(bside.ParallelKalmanFilter(model), data, init_dist, prior=log_prior)
theta_true = torch.cat([A_true.flatten(), torch.tensor([float("-inf"), math.log(sigma)])])  # no true process noise
param_names = ['A11', 'A12', 'A21', 'A22', 'log sd proc', 'log sd meas']

# %% [markdown]
# ## Expectation-maximization
#
# EM gives a point estimate of all of $(A, \Sigma, C, \Gamma)$. A Kalman filter
# and RTS smoother (E-step) alternate with closed-form updates (M-step). $C$ is
# re-estimated too, so $A$ is identifiable only up to a change of state
# coordinates; its eigenvalues are what to compare.

# %%
em_model = bside.SSM(
    xdim=xdim, ydim=ydim,
    dynamics=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(torch.eye(2))), noise_cov=0.1 * torch.eye(2)),
    observations=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(torch.eye(2))), noise_cov=0.1 * torch.eye(2)),
)
log_marg = bside.EM(ssm=em_model, init_dist=init_dist).fit(data, n_iter=500, tol=1e-5)
print(f'EM: {len(log_marg)} iterations, log p(y) {log_marg[0]:.1f} -> {log_marg[-1]:.1f}')
print('eigenvalues of A, true:', torch.linalg.eigvals(A_true))
print('eigenvalues of A, EM:  ', torch.linalg.eigvals(em_model.dynamics.mat_x))

# %% [markdown]
# ## MAP estimate
#
# The filter is differentiable, so `map_estimate` maximizes the log posterior
# with Adam. A batch of starting points `(P, p)` is optimized in one pass.

# %%
starts = torch.cat([torch.eye(2).flatten() + 0.1 * torch.randn(4, 4), torch.full((4, 2), -1.0)], dim=-1)
theta_map, trace = bside.map_estimate(posterior, starts, n_steps=500, lr=0.02)
best = theta_map[trace[-1].argmax()]
for name, true, est in zip(param_names, theta_true, best):
    print(f'{name:>12}: true {true:7.3f}   MAP {est:7.3f}')

plt.figure(figsize=(5, 3))
plt.plot(trace)
plt.xlabel('Adam step')
plt.ylabel('log posterior')
plt.ylim(trace[-1].min() - 50, trace.max() + 5)
plt.show()

# %% [markdown]
# ## Posterior sampling: DRAM and HMC
#
# Both samplers target the same posterior and start from the MAP estimate.
#
# **DRAM** (delayed-rejection adaptive Metropolis) needs only the log density.
# A random walk mixes well only if its proposal matches the posterior's scales
# and correlations. `laplace_covariance` provides them: the inverse Hessian of
# the log posterior at the MAP, computed by autograd through the filter. DRAM
# scales it by $2.4^2/d$, the standard choice, which gives the textbook
# first-stage acceptance of about 25%. DRAM's reported acceptance also counts
# its small second-stage retries after a rejection, so the first-stage rate is
# measured separately here.
#
# **HMC** also uses the gradient and runs four chains as one batch.

# %%
class CountCalls:
    """Wraps the target to count evaluations: each DRAM iteration costs one, plus one per first-stage rejection."""
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, theta):
        self.calls += 1
        return self.f(theta)


n_dram, burn = 6000, 1000
counted = CountCalls(posterior)
dram = DelayedRejectionAdaptiveMetropolis(target=counted, x0=best, cov=bside.laplace_covariance(posterior, best))
counted.calls = 0
with torch.no_grad():
    dram_chain, _ = dram(n_dram)
dram_samples = dram_chain[burn:]
first_stage = 1 - (counted.calls - n_dram) / n_dram
print(f'DRAM: first-stage acceptance {100 * first_stage:.0f}%, overall {100 * dram.acceptance_ratio:.0f}%')

hmc = bside.HMC(posterior, n_leapfrog=12)
hmc_chains, _ = hmc(best.expand(4, 6), n_samples=750, n_warmup=300)   # (750, 4 chains, 6)
hmc_samples = hmc_chains.reshape(-1, 6)
print(f'HMC:  acceptance {100 * hmc.acceptance_rate:.0f}% (4 chains)')

ess_dram = bside.effective_sample_size(dram_samples)
ess_hmc = bside.effective_sample_size(hmc_chains)
print()
print(f'{"":>12} {"true":>8} {"DRAM":>18} {"HMC":>18} {"ESS DRAM":>9} {"ESS HMC":>8}')
for i, name in enumerate(param_names):
    d, h = dram_samples[:, i], hmc_samples[:, i]
    print(f'{name:>12} {theta_true[i]:8.3f} {d.mean():8.3f} +/- {d.std():5.3f} {h.mean():8.3f} +/- {h.std():5.3f}'
          f' {ess_dram[i]:9.0f} {ess_hmc[i]:8.0f}')
print(f'draws: DRAM {len(dram_samples)}, HMC {len(hmc_samples)}')

# %% [markdown]
# Diagnostics: trace plots (DRAM, and one HMC chain), and the autocorrelation
# of each parameter. Good mixing shows traces with no trends or long flat
# stretches, and autocorrelations that decay quickly.

# %%
acf_dram = bside.autocorrelation(dram_samples, max_lag=100)
acf_hmc = bside.autocorrelation(hmc_chains, max_lag=100)
fig, axes = plt.subplots(3, 6, figsize=(16, 6.5))
for i in range(6):
    axes[0, i].plot(dram_samples[:, i], color='C0', lw=0.4)
    axes[0, i].set_title(param_names[i], fontsize=9)
    axes[1, i].plot(hmc_chains[:, 0, i], color='C2', lw=0.4)
    axes[2, i].plot(acf_dram[:, i], color='C0', label='DRAM')
    axes[2, i].plot(acf_hmc[:, i], color='C2', label='HMC')
    axes[2, i].axhline(0, color='gray', lw=0.5)
    axes[2, i].set_xlabel('lag')
axes[0, 0].set_ylabel('DRAM trace')
axes[1, 0].set_ylabel('HMC trace (chain 1)')
axes[2, 0].set_ylabel('autocorrelation')
axes[2, 0].legend(fontsize=8)
fig.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 6, figsize=(15, 2.6))
for i, ax in enumerate(axes):
    ax.hist(dram_samples[:, i], bins=40, density=True, alpha=0.5, color='C0', label='DRAM')
    ax.hist(hmc_samples[:, i], bins=40, density=True, alpha=0.5, color='C2', label='HMC')
    if i != 4:
        ax.axvline(theta_true[i], color='k', lw=1)
    ax.set_title(param_names[i], fontsize=9)
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()

(fig,), (axs,) = SampleVisualizer(hmc_samples).triangular_hist(bins=40, labels=param_names)
for row in range(6):
    for col in range(row + 1):
        if col != 4:
            axs[row, col].axvline(theta_true[col], color='k', lw=0.8)
        if row != col and row != 4:
            axs[row, col].axhline(theta_true[row], color='k', lw=0.8)
fig.set_size_inches(10, 10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Posterior predictive
#
# Push 500 posterior samples of $A$ forward past the end of the data (as in the
# paper's Fig. 6). The band shows where the posterior expects the pendulum to
# be, and it widens as the forecast moves away from the data.

# %%
horizon = 2 * n
draws = hmc_samples[torch.randint(len(hmc_samples), (500,))]
model.update(draws)  # a batch of 500 models
paths = model.dynamics.model.predict(x0.expand(500, 2), T=horizon)
truth = bside.LinearModel(bside.Matrix(A_true)).predict(x0, T=horizon)

t_long = torch.arange(1, horizon + 1) * dt
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for i, (ax, name) in enumerate(zip(axes, ['angle', 'angular velocity'])):
    lo, mid, hi = torch.quantile(paths[..., i], torch.tensor([0.025, 0.5, 0.975]), dim=0)
    ax.fill_between(t_long, lo, hi, color='C1', alpha=0.3, label='95% posterior predictive')
    ax.plot(t_long, mid, 'C1', label='median')
    ax.plot(t_long, truth[:, i], 'k', lw=1, label='truth')
    ax.plot(t, y[:, i], '.', color='C0', label='data')
    ax.axvline(n * dt, color='gray', ls='--', lw=1)
    ax.set_title(name)
    ax.set_xlabel('time')
axes[0].legend(fontsize=8)
plt.show()
