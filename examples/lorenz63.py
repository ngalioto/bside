# %% [markdown]
# # Nonlinear system identification: Lorenz '63
#
# Learn the chaotic Lorenz '63 system from sparse, noisy data, following
# Galioto & Gorodetsky (2020), Sec. 6.6.1: 100 measurements of all three
# states over 10 s, with noise standard deviation 2, starting on the attractor.
#
# - **dynamics**: $x_k = \Psi(x_{k-1}; \theta_{1:3}) + \xi_k$, $\xi_k \sim \mathcal{N}(0, \mathrm{diag}(s_1^2, s_2^2, s_3^2))$,
#   where $\Psi$ integrates the Lorenz ODE over one 0.1 s interval
# - **observations**: $y_k = x_k + \eta_k$, $\eta_k \sim \mathcal{N}(0, s_\eta^2 I)$
#
# There are seven unknowns: $\theta = (\theta_1, \theta_2, \theta_3, \log s_1, \log s_2, \log s_3, \log s_\eta)$.
# With nonlinear dynamics the likelihood $p(y_{1:n} \mid \theta)$ has no closed
# form, so a nonlinear filter approximates it with the predict / update
# recursion from the filtering examples.
#
# Process noise is what makes a chaotic system learnable from 10 s of data: it
# lets the filter correct the trajectory instead of requiring one simulation to
# match all the data. Learning the noise level is delicate, though. Joint
# optimization tends to stop at a high-noise local optimum, where the data are
# explained as noise. The section on process noise shows a continuation strategy
# that avoids this.
#
# Runtime: about half an hour, mostly the optimization and DRAM sections.
# DRAM needs `pip install bside[mcmc]`.

# %%
import math

import torch
import matplotlib.pyplot as plt

import bside
from mcmc_samplers import DelayedRejectionAdaptiveMetropolis, SampleVisualizer

torch.manual_seed(0)

# %% [markdown]
# ## Model and data
#
# The dynamics map is a module whose `update(params)` hook reads $\theta_{1:3}$.
# Written with `unbind(-1)`, it accepts a batch of parameter vectors, so many
# values of $\theta$ can be evaluated in one filter pass. The noise covariances
# are `CholeskyMatrix` objects with diagonal masks: $\theta_{4:6}$ are the
# per-state log standard deviations of the process noise, and $\theta_7$ is
# shared by all three measurement noise variances.

# %%
DT, SUBSTEPS, n = 0.1, 5, 100


class Lorenz63(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def update(self, params):
        if params is not None:
            self.theta = params[..., :3]

    def vector_field(self, x):
        s, r, b = self.theta.unbind(-1)
        x1, x2, x3 = x.unbind(-1)
        return torch.stack([s * (x2 - x1), x1 * (r - x3) - x2, x1 * x2 - b * x3], dim=-1)

    def forward(self, x, u=None):
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):  # RK4
            k1 = self.vector_field(x)
            k2 = self.vector_field(x + h / 2 * k1)
            k3 = self.vector_field(x + h / 2 * k2)
            k4 = self.vector_field(x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


theta_dyn_true = torch.tensor([10.0, 28.0, 8.0 / 3.0])
x0 = torch.tensor([2.0181, 3.5065, 11.8044])
truth_model = bside.NonlinearModel(Lorenz63(theta_dyn_true), in_dim=3, out_dim=3)
x_true = truth_model.predict(x0, T=n)
y = x_true + 2.0 * torch.randn_like(x_true)
data = bside.Data(y=y)
init_dist = bside.FilteringDistribution(mean=x0, cov=torch.eye(3))

diag = torch.eye(3, dtype=torch.bool)
Sigma = bside.CholeskyMatrix(torch.eye(3), mask=diag, indices=torch.tensor([3, 4, 5]))
Gamma = bside.CholeskyMatrix(torch.eye(3), mask=diag, indices=torch.tensor([6, 6, 6]))
model = bside.SSM(
    xdim=3, ydim=3,
    dynamics=bside.NonlinearAdditiveModel(f=Lorenz63(theta_dyn_true.clone()), noise_cov=Sigma, in_dim=3, out_dim=3),
    observations=bside.LinearGaussianModel(model=bside.IdentityModel(3), noise_cov=Gamma),
)


def log_prior(theta, tau=10.0):
    """Flat on theta_1:3; half-normal(tau) on each variance, written for log-std parameters."""
    v = torch.exp(2 * theta[..., 3:])
    return (-0.5 * (v / tau) ** 2 + torch.log(2 * v)).sum(-1)


ukf = bside.UnscentedKalmanFilter(model)
posterior = bside.Posterior(ukf, data, init_dist, prior=log_prior)

t = torch.arange(1, n + 1) * DT
fig = plt.figure(figsize=(12, 3.5))
for i, name in enumerate('xyz'):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.plot(t, x_true[:, i], 'k', lw=1)
    ax.plot(t, y[:, i], '.', color='C0', ms=4)
    ax.set_title(name)
    ax.set_xlabel('time')
ax = fig.add_subplot(1, 4, 4)
ax.plot(x_true[:, 0], x_true[:, 2], 'k', lw=1, label='truth')
ax.plot(y[:, 0], y[:, 2], '.', color='C0', ms=4, label='data')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Likelihood from two filters
#
# Profile $\log p(y \mid \theta)$ over $\theta_2$ ($\rho$), with the other
# dynamics parameters at their true values. Each curve uses a different
# process-noise level. The UKF gives a smooth deterministic approximation; the
# particle filter gives a noisy but unbiased estimate.
#
# The process noise controls how forgiving the likelihood is. With little noise
# the filter must follow the chaotic dynamics closely, so the curve is sharply
# peaked and drops steeply away from the truth. With more noise the filter
# corrects the trajectory at every measurement, and the curve flattens.

# %%
pf = bside.ParticleFilter(model, n_particles=500)
rhos = torch.linspace(22.0, 34.0, 25)
noise_sds = [0.1, 0.3, 1.0, 3.0]
colors = plt.cm.viridis(torch.linspace(0.0, 0.85, len(noise_sds)).numpy())

grid = torch.cat([theta_dyn_true, torch.zeros(3), torch.tensor([math.log(2.0)])]).repeat(len(noise_sds), len(rhos), 1)
grid[..., 1] = rhos
grid[..., 3:6] = torch.log(torch.tensor(noise_sds))[:, None, None]

with torch.no_grad():
    ll_ukf = posterior.log_likelihood(grid)                          # all levels x rhos in one batched UKF pass
    ll_pf = torch.stack([torch.stack([-pf.nlog_marginal_likelihood(data, init_dist, params=theta) for theta in row])
                         for row in grid])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, ll, title in [(axes[0], ll_ukf, 'unscented Kalman filter'), (axes[1], ll_pf, 'particle filter (500 particles)')]:
    for sd, curve, color in zip(noise_sds, ll, colors):
        style = '-' if ax is axes[0] else 'o-'
        ax.plot(rhos, curve, style, color=color, ms=3, lw=1.5 if ax is axes[0] else 0.8, label=f'process sd {sd:g}')
    ax.axvline(28.0, color='k', ls='--', lw=1)
    ax.set_title(title)
    ax.set_xlabel('theta_2 (rho)')
axes[0].set_ylabel('log p(y | theta)')
top = max(ll_ukf.max(), ll_pf.max()).item()
axes[0].set_ylim(top - 400, top + 20)
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Learning the process noise
#
# Start four optimizations from poor guesses of $\theta_{1:3}$, with unit process
# and measurement noise (a batch of four, run in one pass).
#
# **Joint MAP.** Optimizing all seven parameters at once stops at large process
# noise. There the filter attributes the misfit to noise, so the likelihood
# barely responds to the dynamics parameters.
#
# **Continuation** (`noise_continuation`). Fix the process noise at a decreasing
# sequence of levels, fit the remaining parameters at each level starting from
# the previous fit, and record the best log posterior reached at each level.
# Large noise smooths the landscape, so the early fits move the dynamics into
# the right basin; smaller noise then sharpens them. Finally, all parameters are
# freed and refined from the best level.

# %%
starts = torch.tensor([[5.0, 20.0, 1.0], [15.0, 35.0, 4.0], [8.0, 15.0, 2.0], [12.0, 22.0, 5.0]])
theta0 = torch.cat([starts, torch.zeros(4, 3), torch.zeros(4, 1)], dim=-1)

theta_joint, trace = bside.map_estimate(posterior, theta0, n_steps=150, lr=0.05)

levels = [1.0, 0.0, -1.0, -2.0]
profile = bside.noise_continuation(posterior, theta0, noise_indices=[3, 4, 5], levels=levels,
                                   n_steps=50, lr=0.05, final_steps=100)
theta_hat = profile.theta[posterior(profile.theta).argmax()]


def describe(name, theta):
    lp = posterior(theta).item()
    print(f'{name:<13} theta_1:3 = {theta[:3].numpy().round(2)}  process sd = {theta[3:6].exp().numpy().round(2)}  '
          f'measurement sd = {theta[6].exp().item():.2f}  log posterior = {lp:.1f}')


with torch.no_grad():
    for k in range(len(starts)):
        describe(f'joint MAP {k}', theta_joint[k])
    for k in range(len(starts)):
        describe(f'continuation {k}', profile.theta[k])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(trace)
axes[0].set_xlabel('Adam step')
axes[0].set_ylabel('log posterior')
axes[0].set_title('joint MAP (one line per start)')
axes[1].plot(levels, profile.log_posteriors, 'o-')
axes[1].set_xlabel('fixed log process-noise sd')
axes[1].set_ylabel('best log posterior at this level')
axes[1].set_title('continuation profile (one line per start)')
axes[1].invert_xaxis()
fig.tight_layout()
plt.show()

# %% [markdown]
# In this run three of the four starts reach the same low-noise optimum. The
# third (green, initial rho = 15, far from 28) stays in a poor region at every
# fixed noise level, and the final joint refinement only partly recovers it.
# That is still much better than plain joint MAP, but not guaranteed. More steps
# per level, finer levels, or simply more starts (they cost little as a batch)
# make the continuation more reliable.

# %% [markdown]
# ## Posterior sampling with DRAM
#
# Start DRAM at the continuation estimate, with its proposal shaped by the
# Laplace covariance there (the inverse Hessian of the log posterior, from
# autograd through the UKF); DRAM scales it by $2.4^2/d$. DRAM's reported
# acceptance also counts its small second-stage retries after a rejection, so
# the first-stage rate is measured separately: each iteration evaluates the
# target once, plus once more per first-stage rejection.

# %%
class CountCalls:
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, theta):
        self.calls += 1
        return self.f(theta)


n_dram, burn = 2000, 400
counted = CountCalls(posterior)
dram = DelayedRejectionAdaptiveMetropolis(target=counted, x0=theta_hat, cov=bside.laplace_covariance(posterior, theta_hat))
counted.calls = 0
with torch.no_grad():
    chain, _ = dram(n_dram)
samples = chain[burn:]
first_stage = 1 - (counted.calls - n_dram) / n_dram

names = ['theta_1', 'theta_2', 'theta_3', 'log s_1', 'log s_2', 'log s_3', 'log s_eta']
truth = [10.0, 28.0, 8.0 / 3.0, None, None, None, math.log(2.0)]
ess = bside.effective_sample_size(samples)
print(f'first-stage acceptance {100 * first_stage:.0f}%, overall {100 * dram.acceptance_ratio:.0f}%, '
      f'{len(samples)} draws after burn-in')
for i, name in enumerate(names):
    print(f'{name:>9}: {samples[:, i].mean():7.3f} +/- {samples[:, i].std():.3f}   ESS {ess[i]:4.0f}' +
          ('' if truth[i] is None else f'   (true {truth[i]:.3f})'))

# %% [markdown]
# Diagnostics: trace plots (no trends or long flat stretches; true values in
# black) and autocorrelations (quick decay) indicate good mixing.

# %%
acf = bside.autocorrelation(samples, max_lag=100)
fig, axes = plt.subplots(2, 7, figsize=(17, 4.5))
for i in range(7):
    axes[0, i].plot(samples[:, i], color='C0', lw=0.4)
    if truth[i] is not None:
        axes[0, i].axhline(truth[i], color='k', lw=1)
    axes[0, i].set_title(names[i], fontsize=9)
    axes[1, i].plot(acf[:, i], color='C0')
    axes[1, i].axhline(0, color='gray', lw=0.5)
    axes[1, i].set_xlabel('lag')
axes[0, 0].set_ylabel('trace')
axes[1, 0].set_ylabel('autocorrelation')
fig.tight_layout()
plt.show()

# %% [markdown]
# Marginal (diagonal) and pairwise (lower triangle) posterior histograms, with
# the true values in black where they exist. The correlations show which
# parameters the data can only pin down jointly.

# %%
(fig,), (axs,) = SampleVisualizer(samples).triangular_hist(bins=30, labels=names)
for row in range(7):
    for col in range(row + 1):
        if truth[col] is not None:
            axs[row, col].axvline(truth[col], color='k', lw=0.8)
        if row != col and truth[row] is not None:
            axs[row, col].axhline(truth[row], color='k', lw=0.8)
fig.set_size_inches(11, 11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Prediction
#
# Simulate the learned dynamics (posterior mean of $\theta_{1:3}$) from the same
# initial condition. The trajectories separate after a few seconds, as they must
# for a chaotic system. The long-run behaviour is what the data can identify:
# the learned model reproduces the attractor (compare the paper's Fig. 18).

# %%
theta_mean = samples[:, :3].mean(0)
learned = bside.NonlinearModel(Lorenz63(theta_mean), in_dim=3, out_dim=3)
T_long = 2000  # 200 s
with torch.no_grad():
    true_long = truth_model.predict(x0, T=T_long)
    learned_long = learned.predict(x0, T=T_long)

t_long = torch.arange(1, T_long + 1) * DT
fig = plt.figure(figsize=(14, 3.6))
ax = fig.add_subplot(1, 4, (1, 2))
window = t_long <= 20
ax.plot(t_long[window], true_long[window, 0], 'k', lw=1, label='truth')
ax.plot(t_long[window], learned_long[window, 0], 'C1', lw=1, label='learned model')
ax.plot(t, y[:, 0], '.', color='C0', ms=4, label='data')
ax.axvline(n * DT, color='gray', ls='--', lw=1)
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.legend(fontsize=8, loc='upper right')
for k, (traj, color, title) in enumerate([(true_long, 'k', 'truth over 200 s, with data'),
                                          (learned_long, 'C1', 'learned model over 200 s')]):
    ax = fig.add_subplot(1, 4, 3 + k, sharex=ax if k else None, sharey=ax if k else None)
    ax.plot(traj[:, 0], traj[:, 2], color=color, lw=0.3, alpha=0.6)
    if k == 0:
        ax.plot(y[:, 0], y[:, 2], '.', color='C0', ms=4)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(title, fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Comparison with methods that ignore process noise
#
# Two standard alternatives fit the same data without a process-noise model:
#
# - **Derivative regression** (SINDy with the true library of Lorenz terms).
#   Estimate dx/dt by central differences of the noisy data, then solve for
#   $\theta$ by least squares. With measurement noise of 2 and spacing 0.1 s,
#   the derivative estimates are very noisy, and using noisy states as
#   regressors biases the estimates.
# - **Deterministic fit** (output error, single shooting). Simulate the Lorenz
#   ODE from $x_0$ and minimize the squared error to the data over $\theta$,
#   from the same four starting points. Without process noise the simulation
#   must match all 10 s at once, which is the rough, chaotic landscape of the
#   paper's Fig. 1.

# %%
# Derivative regression with the true library
dy = (y[2:] - y[:-2]) / (2 * DT)
x1, x2, x3 = y[1:-1].unbind(-1)
theta_reg = torch.stack([
    torch.linalg.lstsq((x2 - x1)[:, None], dy[:, 0:1]).solution.squeeze(),
    torch.linalg.lstsq(x1[:, None], (dy[:, 1] + x1 * x3 + x2)[:, None]).solution.squeeze(),
    torch.linalg.lstsq(x3[:, None], (x1 * x2 - dy[:, 2])[:, None]).solution.squeeze(),
])

# Deterministic fit from the same four starts (batched)
shooting_model = bside.NonlinearModel(Lorenz63(theta_dyn_true.clone()), in_dim=3, out_dim=3)


def neg_squared_error(theta):
    shooting_model.update(theta)
    simulated = shooting_model.predict(x0.expand(*theta.shape[:-1], 3), T=n)
    return -0.5 * ((simulated - y) ** 2).sum((-1, -2)) / 4.0   # Gaussian log-likelihood, sd 2


theta_det, det_trace = bside.map_estimate(neg_squared_error, starts, n_steps=300, lr=0.05)
best_det = theta_det[det_trace[-1].argmax()]

estimates = {
    'Bayes (posterior mean)': theta_mean,
    'derivative regression': theta_reg,
    'deterministic fit (best start)': best_det,
}
print(f'{"":>31} {"theta_1":>8} {"theta_2":>8} {"theta_3":>8}')
print(f'{"truth":>31} {10.0:8.3f} {28.0:8.3f} {8 / 3:8.3f}')
for name, theta in estimates.items():
    print(f'{name:>31} ' + ' '.join(f'{v:8.3f}' for v in theta.tolist()))
print('deterministic fit, all four starts:\n', theta_det.numpy().round(2))

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharex=True, sharey=True)
panels = [('truth', true_long, 'k')] + [
    (name, bside.NonlinearModel(Lorenz63(theta), in_dim=3, out_dim=3).predict(x0, T=T_long), color)
    for (name, theta), color in zip(estimates.items(), ['C1', 'C3', 'C4'])
]
with torch.no_grad():
    for ax, (name, traj, color) in zip(axes, panels):
        ax.plot(traj[:, 0], traj[:, 2], color=color, lw=0.3, alpha=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('x')
axes[0].set_ylabel('z')
fig.tight_layout()
plt.show()

# %% [markdown]
# Only the filter-based posterior recovers all three parameters. Derivative
# regression keeps the attractor's shape but underestimates theta_1 (about 6
# instead of 10), the bias from noisy derivatives and noisy regressors. The
# deterministic fit stops at a different local optimum from each start, all far
# from the truth; its best start collapses to a fixed point. This is the rough,
# multi-modal landscape that process noise smooths out.
