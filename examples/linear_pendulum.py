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
def linear_model():
    """Learnable linear model: theta = (A11, A12, A21, A22, log sd proc, log sd meas)."""
    diag = torch.eye(2, dtype=torch.bool)
    A = bside.Matrix(torch.eye(2), mask=torch.ones(2, 2, dtype=torch.bool), indices=torch.arange(4))
    Sigma = bside.CholeskyMatrix(0.01 * torch.eye(2), mask=diag, indices=torch.tensor([4, 4]))
    Gamma = bside.CholeskyMatrix(0.01 * torch.eye(2), mask=diag, indices=torch.tensor([5, 5]))
    return bside.SSM(
        xdim=xdim, ydim=ydim,
        dynamics=bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=Sigma),
        observations=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(torch.eye(2))), noise_cov=Gamma),
    )


model = linear_model()


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
em = bside.EM(ssm=em_model, init_dist=init_dist)
log_marg = em.fit(data, n_iter=500, tol=1e-5)
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
    if torch.isfinite(theta_true[i]):
        for row in (0, 1):
            axes[row, i].axhline(theta_true[i], color='k', lw=1)
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
# ## Comparison with DMD, TDMD and EM
#
# The paper compares the Bayesian approach with dynamic mode decomposition.
# **DMD** regresses each measurement on the previous one by least squares,
# which treats the measurement noise as process noise. **TDMD** uses total
# least squares, which allows noise in both. **EM** is the maximum-likelihood
# point estimate from above. All of them, and 500 posterior samples of $A$, are
# simulated forward from the same initial condition, past the end of the data
# (as in the paper's Fig. 6).

# %%
horizon = 2 * n
truth = bside.LinearModel(bside.Matrix(A_true)).predict(x0, T=horizon)

draws = hmc_samples[torch.randint(len(hmc_samples), (500,))]
model.update(draws)  # a batch of 500 models
paths = model.dynamics.model.predict(x0.expand(500, 2), T=horizon)

dmd, tdmd = bside.DMD(), bside.TDMD()
dmd.fit(data)
tdmd.fit(data)
point = {
    'DMD': (dmd.A, bside.LinearModel(bside.Matrix(dmd.A)).predict(x0, T=horizon)),
    'TDMD': (tdmd.A, bside.LinearModel(bside.Matrix(tdmd.A)).predict(x0, T=horizon)),
    'EM': (em_model.dynamics.mat_x,
           em_model.dynamics.model.predict(em.init_dist.mean, T=horizon) @ em_model.observations.mat_x.mT),
    'Bayes (posterior mean)': (draws[:, :4].mean(0).reshape(2, 2), paths.mean(0)),
}


def rmse(pred, window):
    return (pred[window] - truth[window]).pow(2).mean().sqrt().item()


fit_window, forecast_window = slice(0, n), slice(n, horizon)
print(f'{"":>24} {"|eig| (true " + format(torch.linalg.eigvals(A_true).abs()[0].item(), ".4f") + ")":>20} '
      f'{"RMSE 0-4 s":>11} {"RMSE 4-8 s":>11}')
for name, (A_est, pred) in point.items():
    modulus = torch.linalg.eigvals(A_est).abs().max().item()
    print(f'{name:>24} {modulus:20.4f} {rmse(pred, fit_window):11.4f} {rmse(pred, forecast_window):11.4f}')

# %%
t_long = torch.arange(1, horizon + 1) * dt
styles = {'DMD': 'C3', 'TDMD': 'C4', 'EM': 'C5'}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for i, (ax, name) in enumerate(zip(axes, ['angle', 'angular velocity'])):
    lo, hi = torch.quantile(paths[..., i], torch.tensor([0.025, 0.975]), dim=0)
    ax.fill_between(t_long, lo, hi, color='C1', alpha=0.25, label='Bayes 95% predictive')
    ax.plot(t_long, paths[..., i].mean(0), 'C1', lw=1.5, label='Bayes mean')
    for method, color in styles.items():
        ax.plot(t_long, point[method][1][:, i], color=color, lw=1, label=method)
    ax.plot(t_long, truth[:, i], 'k', lw=1, label='truth')
    ax.plot(t, y[:, i], '.', color='C0', label='data')
    ax.axvline(n * dt, color='gray', ls='--', lw=1)
    ax.set_title(name)
    ax.set_xlabel('time')
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
plt.show()

# %% [markdown]
# DMD treats all noise as process noise, so its eigenvalues shrink toward zero
# and its forecast decays to rest; TDMD damps less but drifts out of phase, as
# in the paper. Both the Bayesian posterior and EM recover an essentially
# undamped oscillator. With 40 points, EM's point estimate forecasts best here.
# The posterior mean averages over all plausible matrices, including slightly
# damped ones, so its oscillation shrinks with time. What the posterior adds is
# the band: it widens past the data and contains the truth throughout.

# %% [markdown]
# ## The noise trade-off, and model error
#
# Process noise and measurement noise compete to explain the same residuals:
# either the model is imperfect and the data are accurate, or the model is
# right and the data are noisy. The paper (Sec. 6.3) notes their variances are
# negatively correlated, and that a linear model of a *nonlinear* pendulum can
# give a bimodal noise posterior (Sec. 6.4, Fig. 12a).
#
# To check both, compare the data set above with the paper's model-error case.
# Its data come from the nonlinear pendulum with $x_0 = (2.5, 0)$: 24 points
# over 4 s, noise standard deviation 0.1, fit with the same linear model.
# Bimodality is easy to miss with one chain, so each case runs four HMC chains
# in parallel. For the model-error case, the chains start from the optima of
# four MAP fits that begin at very different noise levels.

# %%
def nonlinear_pendulum(x0, steps, step_dt, substeps=20):
    """RK4 integration of the nonlinear pendulum, sampled every step_dt."""
    def vector_field(x):
        return torch.stack([x[..., 1], -g * torch.sin(x[..., 0])], dim=-1)
    out, x, h = [], x0, step_dt / substeps
    for _ in range(steps):
        for _ in range(substeps):
            k1 = vector_field(x)
            k2 = vector_field(x + h / 2 * k1)
            k3 = vector_field(x + h / 2 * k2)
            k4 = vector_field(x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x)
    return torch.stack(out)


x0_nl = torch.tensor([2.5, 0.0])
x_nl = nonlinear_pendulum(x0_nl, 24, 4.0 / 24)
y_nl = x_nl + 0.1 * torch.randn_like(x_nl)
posterior_nl = bside.Posterior(bside.ParallelKalmanFilter(linear_model()), bside.Data(y=y_nl),
                               bside.FilteringDistribution(mean=x0_nl, cov=0.01 * torch.eye(2)), prior=log_prior)

noise_starts = torch.log(torch.tensor([[0.01, 0.3], [0.3, 0.01], [0.1, 0.1], [0.03, 0.03]]))
starts_nl = torch.cat([torch.eye(2).flatten() + 0.1 * torch.randn(4, 4), noise_starts], dim=-1)
map_nl, trace_nl = bside.map_estimate(posterior_nl, starts_nl, n_steps=600, lr=0.02)
print('Model error: MAP from four noise starts')
for k, (theta, lp) in enumerate(zip(map_nl, trace_nl[-1])):
    print(f'  start {k + 1}: process sd {theta[4].exp():.3f}, measurement sd {theta[5].exp():.3f}, '
          f'log posterior {lp:.1f}')

hmc_nl = bside.HMC(posterior_nl, n_leapfrog=10)
chains_nl, _ = hmc_nl(map_nl, n_samples=800, n_warmup=400)

for name, chains in [('linear truth (above)', hmc_chains), ('model error', chains_nl)]:
    flat = chains.reshape(-1, 6)
    variances = torch.exp(2 * flat[:, 4:])
    print()
    print(f'{name}: R-hat {bside.potential_scale_reduction(chains).numpy().round(3)}')
    print(f'  corr(process var, measurement var) = {torch.corrcoef(variances.T)[0, 1]:.2f}, '
          f'corr(log sds) = {torch.corrcoef(flat[:, 4:].T)[0, 1]:.2f}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (title, chains, optima) in zip(axes, [('linear truth', hmc_chains, None),
                                              ('model error (paper Sec. 6.4)', chains_nl, map_nl)]):
    for c in range(chains.shape[1]):
        v = torch.exp(2 * chains[:, c, 4:])
        ax.plot(v[:, 0], v[:, 1], '.', ms=1.5, alpha=0.4, color=f'C{c}')
        ax.plot([], [], 'o', ms=4, color=f'C{c}', label=f'chain {c + 1}')   # legend entry
    if optima is not None:
        v = torch.exp(2 * optima[:, 4:])
        ax.plot(v[:, 0], v[:, 1], 'k*', ms=10, label='MAP optima')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('process-noise variance')
    ax.set_ylabel('measurement-noise variance')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
fig.tight_layout()
plt.show()

# %% [markdown]
# In both cases the process- and measurement-noise variances are only weakly
# negatively correlated, and the four chains agree (R-hat near 1). With the
# linear truth the process noise sits far below the measurement noise, so the
# two barely compete. With model error, MAP fits started at different noise
# levels stop at different optima, the paper's two explanations of the data.
# The chains, however, all settle around the high-process-noise optimum: the
# low-process-noise optima have log posteriors more than 10 lower and carry
# negligible posterior mass for this data set. Whether a second mode carries
# visible mass depends on the data realization and noise level; parallel
# chains started from dispersed optima are how to find out.
