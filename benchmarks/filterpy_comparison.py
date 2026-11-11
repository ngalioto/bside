"""
Accuracy and speed of bside's filters against filterpy (NumPy reference).

    pip install filterpy
    python benchmarks/filterpy_comparison.py

Kalman filter on a linear pendulum and unscented Kalman filter on a nonlinear
pendulum, both with the angle measured; filterpy runs in float64 NumPy, bside
in float64 torch.  Reports the largest difference in filtered means and in the
log-likelihood, and wall-clock times (single trajectory, and bside on a batch).
"""

import time

import numpy as np
import torch
from filterpy.kalman import KalmanFilter as FPKalman, MerweScaledSigmaPoints, UnscentedKalmanFilter as FPUnscented

import bside

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

dt, g, qc, r = 0.01, 9.81, 0.1, 0.1
Q = qc * torch.tensor([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
R = torch.tensor([[r**2]])
x0, P0 = torch.tensor([1.5, 0.0]), 0.1 * torch.eye(2)
init = bside.FilteringDistribution(mean=x0, cov=P0)


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def report(name, d_mean, d_ll, times):
    cols = '   '.join(f'{k} {v:7.3f}s' for k, v in times.items())
    print(f'{name:<5} max|dm| {d_mean:.1e}  |dll| {d_ll:.1e}   {cols}')


# ---------------------------------------------------------------------------
# Linear: Kalman filter
# ---------------------------------------------------------------------------

A = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt)
H = torch.tensor([[1.0, 0.0]])
linear = bside.SSM(xdim=2, ydim=1,
                   dynamics=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(A)), noise_cov=Q),
                   observations=bside.LinearGaussianModel(model=bside.LinearModel(bside.Matrix(H)), noise_cov=R))


def filterpy_kf(y):
    kf = FPKalman(dim_x=2, dim_z=1)
    kf.x, kf.P, kf.F, kf.H, kf.Q, kf.R = x0.numpy().copy(), P0.numpy().copy(), A.numpy(), H.numpy(), Q.numpy(), R.numpy()
    means, ll = [], 0.0
    for yt in y:
        kf.predict()
        kf.update(yt)
        means.append(kf.x.copy())
        ll += kf.log_likelihood
    return np.array(means), ll


print('Kalman filter (linear pendulum)')
for T in (1_000, 10_000):
    y = linear.measure(x=x0, T=T)
    data = bside.Data(y=y)
    (m_fp, ll_fp), t_fp = timed(lambda: filterpy_kf(y.numpy()))
    (h, ll), t_kf = timed(lambda: bside.KalmanFilter(linear).filter(data, init, return_history=True, compute_log_prob=True))
    (hp, llp), t_par = timed(lambda: bside.ParallelKalmanFilter(linear).filter(data, init, return_history=True, compute_log_prob=True))
    m = torch.stack([d.mean for d in h[1:]]).numpy()
    mp = torch.stack([d.mean for d in hp[1:]]).numpy()
    batch = bside.Data(y=torch.stack([linear.measure(x=x0, T=T) for _ in range(100)]))
    _, t_batch = timed(lambda: bside.ParallelKalmanFilter(linear).filter(batch, init, compute_log_prob=True))
    report(f'T={T}', max(np.abs(m - m_fp).max(), np.abs(mp - m_fp).max()),
           max(abs(ll.item() - ll_fp), abs(llp.item() - ll_fp)),
           {'filterpy': t_fp, 'bside': t_kf, 'parallel': t_par, 'parallel x100': t_batch})


# ---------------------------------------------------------------------------
# Nonlinear: unscented Kalman filter
# ---------------------------------------------------------------------------

def step(x, u=None):
    return torch.stack([x[..., 0] + dt * x[..., 1], x[..., 1] - dt * g * torch.sin(x[..., 0])], dim=-1)


def sensor(x, u=None):
    return torch.sin(x[..., :1])


nonlinear = bside.SSM(xdim=2, ydim=1,
                      dynamics=bside.NonlinearAdditiveModel(f=step, noise_cov=Q, in_dim=2, out_dim=2),
                      observations=bside.NonlinearAdditiveModel(f=sensor, noise_cov=R, in_dim=2, out_dim=1))


def filterpy_ukf(y):
    points = MerweScaledSigmaPoints(n=2, alpha=1.0, beta=2.0, kappa=0.0)
    ukf = FPUnscented(dim_x=2, dim_z=1, dt=dt, points=points,
                      fx=lambda x, dt: step(torch.from_numpy(x)).numpy(),
                      hx=lambda x: sensor(torch.from_numpy(x)).numpy())
    ukf.x, ukf.P, ukf.Q, ukf.R = x0.numpy().copy(), P0.numpy().copy(), Q.numpy(), R.numpy()
    means, ll = [], 0.0
    for yt in y:
        ukf.predict()
        ukf.update(yt)
        means.append(ukf.x.copy())
        ll += ukf.log_likelihood
    return np.array(means), ll


print('Unscented Kalman filter (nonlinear pendulum)')
T = 1_000
y = nonlinear.measure(x=x0, T=T)
(m_fp, ll_fp), t_fp = timed(lambda: filterpy_ukf(y.numpy()))
(h, ll), t_ukf = timed(lambda: bside.UnscentedKalmanFilter(nonlinear).filter(
    bside.Data(y=y), init, return_history=True, compute_log_prob=True))
batch = bside.Data(y=torch.stack([nonlinear.measure(x=x0, T=T) for _ in range(100)]))
_, t_batch = timed(lambda: bside.UnscentedKalmanFilter(nonlinear).filter(batch, init, compute_log_prob=True))
m = torch.stack([d.mean for d in h[1:]]).numpy()
report(f'T={T}', np.abs(m - m_fp).max(), abs(ll.item() - ll_fp), {'filterpy': t_fp, 'bside': t_ukf, 'bside x100': t_batch})
