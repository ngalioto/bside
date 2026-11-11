# %% [markdown]
# # Linear filtering
#
# State estimation for a pendulum under the small-angle (linear) approximation.
# The state-space model is built from its two halves, each a deterministic map
# plus additive Gaussian noise:
#
# - **dynamics**: $x_k = A x_{k-1} + \xi_k,\quad \xi_k \sim \mathcal{N}(0, \Sigma)$
# - **observations**: $y_k = C x_k + \eta_k,\quad \eta_k \sim \mathcal{N}(0, \Gamma)$
#
# Methods shown: Kalman filter, square-root Kalman filter, RTS smoother, their
# parallel-in-time versions, irregularly sampled data, and filtering a batch of
# trajectories at once.

# %%
import time

import torch
import matplotlib.pyplot as plt

import bside

torch.manual_seed(0)

# %% [markdown]
# ## Model and data
#
# The state is (angle, angular velocity); only the angle is measured.

# %%
dt, g, qc, r = 0.1, 9.81, 0.1, 0.1
T = 60
xdim, ydim = 2, 1

A = bside.Matrix(torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt))
Sigma = bside.PSDMatrix(qc * torch.tensor([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]))
C = bside.Matrix(torch.tensor([[1.0, 0.0]]))
Gamma = bside.PSDMatrix(torch.tensor([[r**2]]))

dynamics = bside.LinearGaussianModel(model=bside.LinearModel(A), noise_cov=Sigma)
observations = bside.LinearGaussianModel(model=bside.LinearModel(C), noise_cov=Gamma)
ssm = bside.SSM(xdim=xdim, ydim=ydim, dynamics=dynamics, observations=observations)

x0 = torch.tensor([1.5, 0.0])
x_true, y = ssm.measure(x=x0, T=T, return_x=True)  # x_1..x_T and y_1..y_T
x_true = torch.cat([x0[None], x_true])              # prepend x_0 to align with the filter history
data = bside.Data(y=y)
init_dist = bside.FilteringDistribution(mean=x0, cov=0.1 * torch.eye(xdim))

t = torch.arange(T + 1) * dt
plt.figure(figsize=(8, 3))
plt.plot(t[1:], y, '.', label='measurements')
plt.plot(t, x_true[:, 0], 'k', lw=1, label='true angle')
plt.xlabel('time')
plt.legend()
plt.show()

# %% [markdown]
# ## Kalman filter and square-root Kalman filter
#
# Both run the same predict (dynamics) / update (observations) recursion and
# accumulate the marginal likelihood $\log p(y_{1:T})$. The square-root filter
# propagates Cholesky factors only, which is more robust when covariances are
# ill-conditioned; on a well-conditioned problem the two agree.

# %%
kf = bside.KalmanFilter(ssm)
kf_hist, kf_ll = kf.filter(data, init_dist, return_history=True, compute_log_prob=True)

srkf = bside.SquareRootKalmanFilter(ssm)
sr_hist, sr_ll = srkf.filter(data, init_dist, return_history=True, compute_log_prob=True)

kf_m, kf_P = bside.collate_filtering_distributions(kf_hist)
sr_m, sr_P = bside.collate_filtering_distributions(sr_hist)
print(f'log p(y), Kalman:      {kf_ll.item():.3f}')
print(f'log p(y), square root: {sr_ll.item():.3f}')
print(f'max |mean difference|: {(kf_m - sr_m).abs().max().item():.2e}')

# %% [markdown]
# ## RTS smoother
#
# The smoother runs backward over the filter history to give $p(x_k \mid y_{1:T})$,
# using every measurement for every state.

# %%
rts_hist = bside.RTSSmoother(ssm).smooth(kf_hist)
rts_m, rts_P = bside.collate_filtering_distributions(rts_hist)


def rmse(m):
    return (m - x_true).pow(2).mean().sqrt().item()


print(f'RMSE  filter: {rmse(kf_m):.4f}   smoother: {rmse(rts_m):.4f}')

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for i, (ax, name) in enumerate(zip(axes, ['angle', 'angular velocity'])):
    ax.plot(t, x_true[:, i], 'k', lw=1, label='true')
    for m, P, label, color in [(kf_m, kf_P, 'Kalman filter', 'C0'), (rts_m, rts_P, 'RTS smoother', 'C1')]:
        sd = P[:, i, i].sqrt()
        ax.plot(t, m[:, i], color=color, label=label)
        ax.fill_between(t, m[:, i] - 2 * sd, m[:, i] + 2 * sd, color=color, alpha=0.2)
    ax.set_ylabel(name)
axes[0].legend()
axes[-1].set_xlabel('time')
plt.show()

# %% [markdown]
# ## Batched filtering
#
# Data with shape `(B, T, ydim)` are treated as `B` independent trajectories and
# filtered in one pass, giving one log-likelihood per trajectory.

# %%
B = 20
ys = torch.stack([ssm.measure(x=x0, T=T) for _ in range(B)])
batch = bside.Data(y=ys)

start = time.perf_counter()
_, ll = kf.filter(batch, init_dist, compute_log_prob=True)
batched_time = time.perf_counter() - start

start = time.perf_counter()
ll_loop = torch.stack([kf.filter(bside.Data(y=ys[b]), init_dist, compute_log_prob=True)[1] for b in range(B)])
loop_time = time.perf_counter() - start

print(f'log-likelihoods: shape {tuple(ll.shape)}, max difference from loop {(ll - ll_loop).abs().max().item():.1e}')
print(f'batched: {batched_time:.2f} s   loop: {loop_time:.2f} s')

# %% [markdown]
# ## Parallel-in-time filtering and smoothing
#
# `ParallelKalmanFilter` and `ParallelRTSSmoother` compute the same estimates as
# a prefix scan over time (Sarkka & Garcia-Fernandez, 2021): about log2(T)
# vectorized passes instead of $T$ sequential steps.

# %%
long = bside.Data(y=ssm.measure(x=x0, T=10_000))
for flt in (kf, bside.ParallelKalmanFilter(ssm)):
    start = time.perf_counter()
    hist, ll_long = flt.filter(long, init_dist, return_history=True, compute_log_prob=True)
    print(f'{type(flt).__name__:<22} T=10000: {time.perf_counter() - start:.2f} s, log p(y) = {ll_long.item():.2f}')

par_m, _ = bside.collate_filtering_distributions(bside.ParallelRTSSmoother(ssm).smooth(kf_hist))
print(f'max |RTS - parallel RTS| on the first data set: {(par_m - rts_m).abs().max().item():.1e}')

# %% [markdown]
# ## Irregularly sampled data
#
# Measurements need not arrive at every step. `Data(y, mask=...)` takes a mask
# that is 1 where a measurement is available and 0 elsewhere; at unmeasured
# times the filter only predicts, so the uncertainty grows until the next
# measurement arrives. The values of `y` there are ignored.

# %%
mask = torch.zeros(T, dtype=torch.bool)
mask[torch.cumsum(torch.randint(1, 7, (T,)), 0)[:T].clamp(max=T - 1)] = True  # random gaps of 1-6 steps
sparse = bside.Data(y=y, mask=mask)

sparse_hist = kf.filter(sparse, init_dist, return_history=True)
sparse_m, sparse_P = bside.collate_filtering_distributions(sparse_hist)
smooth_m, smooth_P = bside.collate_filtering_distributions(bside.RTSSmoother(ssm).smooth(sparse_hist))
print(f'{int(mask.sum())} of {T} time points measured')

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t, x_true[:, 0], 'k', lw=1, label='true angle')
ax.plot(t[1:][mask], y[mask], 'o', ms=3, color='k', label='measurements')
for m, P, label, color in [(sparse_m, sparse_P, 'filter', 'C0'), (smooth_m, smooth_P, 'smoother', 'C1')]:
    sd = P[:, 0, 0].sqrt()
    ax.plot(t, m[:, 0], color=color, label=label)
    ax.fill_between(t, m[:, 0] - 2 * sd, m[:, 0] + 2 * sd, color=color, alpha=0.2)
ax.set_xlabel('time')
ax.legend(ncol=4, fontsize=8)
plt.show()
