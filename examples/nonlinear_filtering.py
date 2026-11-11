# %% [markdown]
# # Nonlinear filtering
#
# State estimation for a full (nonlinear) pendulum measured through a nonlinear
# sensor. As in the linear case, the model is built from its two halves, each a
# deterministic map plus additive Gaussian noise:
#
# - **dynamics**: $x_k = \Psi(x_{k-1}) + \xi_k,\quad \xi_k \sim \mathcal{N}(0, \Sigma)$
# - **observations**: $y_k = h(x_k) + \eta_k,\quad \eta_k \sim \mathcal{N}(0, \Gamma)$
#
# Methods shown: unscented, Gauss-Hermite and cubature Kalman filters, the
# ensemble Kalman filter, the bootstrap particle filter, and the unscented RTS
# and particle smoothers.

# %%
import torch
import matplotlib.pyplot as plt

import bside

torch.manual_seed(0)

# %% [markdown]
# ## Model and data
#
# $\Psi$ is an explicit-Euler step of $\ddot\theta = -g \sin\theta$, and the
# sensor measures $h(x) = \sin\theta$.

# %%
dt, g, qc, r = 0.01, 9.81, 0.1, 0.1
T = 500
xdim, ydim = 2, 1


def pendulum_step(x, u=None):
    theta, omega = x[..., 0], x[..., 1]
    return torch.stack([theta + dt * omega, omega - dt * g * torch.sin(theta)], dim=-1)


def sensor(x, u=None):
    return torch.sin(x[..., :1])


Sigma = bside.PSDMatrix(qc * torch.tensor([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]))
Gamma = bside.PSDMatrix(torch.tensor([[r**2]]))

dynamics = bside.NonlinearAdditiveModel(f=pendulum_step, noise_cov=Sigma, in_dim=xdim, out_dim=xdim)
observations = bside.NonlinearAdditiveModel(f=sensor, noise_cov=Gamma, in_dim=xdim, out_dim=ydim)
ssm = bside.SSM(xdim=xdim, ydim=ydim, dynamics=dynamics, observations=observations)

x0 = torch.tensor([1.5, 0.0])
x_true, y = ssm.measure(x=x0, T=T, return_x=True)  # x_1..x_T and y_1..y_T
x_true = torch.cat([x0[None], x_true])              # prepend x_0 to align with the filter history
data = bside.Data(y=y)
init_dist = bside.FilteringDistribution(mean=x0, cov=0.1 * torch.eye(xdim))

t = torch.arange(T + 1) * dt
plt.figure(figsize=(8, 3))
plt.plot(t[1:], y, '.', ms=2, label='measurements of sin(angle)')
plt.plot(t, x_true[:, 0], 'k', lw=1, label='true angle')
plt.xlabel('time')
plt.legend()
plt.show()

# %% [markdown]
# ## Filters
#
# Every filter runs the same predict / update recursion; they differ only in how
# a distribution is pushed through a nonlinear map:
#
# - **Gaussian filters** (unscented, Gauss-Hermite, cubature) propagate
#   deterministic quadrature points.
# - **Ensemble Kalman** propagates random samples with a Gaussian update.
# - **Particle filter** propagates weighted samples; its likelihood estimate
#   is unbiased.

# %%
filters = {
    'Unscented': bside.UnscentedKalmanFilter(ssm),
    'Gauss-Hermite': bside.GaussHermiteFilter(ssm, order=5),
    'Cubature': bside.CubatureKalmanFilter(ssm),
    'Ensemble Kalman': bside.EnsembleKalmanFilter(ssm, ensemble_size=200),
    'Particle': bside.ParticleFilter(ssm, n_particles=1000),
}

histories, estimates = {}, {}
print(f'{"filter":<16} {"log p(y)":>9} {"RMSE":>8}')
for name, flt in filters.items():
    hist, ll = flt.filter(data, init_dist, return_history=True, compute_log_prob=True)
    histories[name] = hist
    estimates[name] = bside.collate_filtering_distributions(hist)
    err = (estimates[name][0] - x_true).pow(2).mean().sqrt()
    print(f'{name:<16} {ll.item():9.2f} {err.item():8.4f}')

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for i, (ax, label) in enumerate(zip(axes, ['angle', 'angular velocity'])):
    ax.plot(t, x_true[:, i], 'k', lw=2, label='true')
    for name, (m, _) in estimates.items():
        ax.plot(t, m[:, i], lw=1, label=name)
    ax.set_ylabel(label)
axes[0].legend(ncol=3, fontsize=8)
axes[-1].set_xlabel('time')
plt.show()

# %% [markdown]
# ## Smoothers
#
# The unscented RTS smoother runs backward over the unscented filter's history.
# The particle smoother (forward filtering, backward simulation) draws whole
# trajectories from the particle filter's history.

# %%
urts_hist = bside.UnscentedRTSSmoother(ssm).smooth(histories['Unscented'])
ps_hist = bside.ParticleSmoother(ssm, n_samples=200).smooth(histories['Particle'])

smoothed = {
    'Unscented RTS': bside.collate_filtering_distributions(urts_hist),
    'Particle smoother': bside.collate_filtering_distributions(ps_hist),
}
for name, (m, _) in smoothed.items():
    print(f'{name:<18} RMSE {(m - x_true).pow(2).mean().sqrt().item():.4f}')

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for i, (ax, label) in enumerate(zip(axes, ['angle', 'angular velocity'])):
    ax.plot(t, x_true[:, i], 'k', lw=1, label='true')
    for (name, (m, P)), color in zip(smoothed.items(), ['C0', 'C4']):
        sd = P[:, i, i].sqrt()
        ax.plot(t, m[:, i], color=color, label=name)
        ax.fill_between(t, m[:, i] - 2 * sd, m[:, i] + 2 * sd, color=color, alpha=0.2)
    ax.set_ylabel(label)
axes[0].legend()
axes[-1].set_xlabel('time')
plt.show()
