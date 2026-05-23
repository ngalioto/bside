"""
Linear-Gaussian smoothing example.

Runs a Kalman filter forward and an RTS smoother backward on a damped
harmonic-oscillator trajectory observed only through its position.  The smoother
posterior should hug the true state much more tightly than the forward filter
alone.

Run from the repo root::

    python examples/linear_smoother.py
"""

import torch
import matplotlib.pyplot as plt

import bside


def main() -> None:
    dt = 0.05
    g = 9.81
    qc = 0.05
    r = 0.2
    T = 200
    xdim, ydim = 2, 1

    # ---- True system ----
    A = torch.linalg.matrix_exp(torch.tensor([[0.0, 1.0], [-g, 0.0]]) * dt)
    Q = torch.tensor([
        [qc * dt**3 / 3, qc * dt**2 / 2],
        [qc * dt**2 / 2, qc * dt],
    ])
    C = torch.tensor([[1.0, 0.0]])
    R = torch.tensor([[r ** 2]])

    dynamics = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(A)),
        noise_cov=bside.PSDMatrix(Q),
    )
    observations = bside.LinearGaussianModel(
        model=bside.LinearModel(bside.Matrix(C)),
        noise_cov=bside.PSDMatrix(R),
    )
    ssm = bside.SSM(xdim=xdim, ydim=ydim, dynamics=dynamics, observations=observations)

    # ---- Generate data ----
    x0 = torch.tensor([1.5, 0.0])
    x_true, y = ssm.measure(x=x0, T=T, keep_y0=True, return_x=True)
    data = bside.Data(y=y)
    init_dist = bside.FilteringDistribution(mean=x0, cov=0.01 * torch.eye(xdim))

    # ---- Forward filter ----
    kf = bside.KalmanFilter(model=ssm)
    filt_hist = kf.filter(data=data, init_dist=init_dist, y0=True, return_history=True)

    # ---- Backward smoother ----
    rts = bside.RTSSmoother(model=ssm)
    smooth_hist = rts.smooth(filt_hist)

    # ---- Plot ----
    t = torch.arange(T + 1) * dt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    labels = ["position", "velocity"]
    for ii, ax in enumerate(axes):
        x_filt = torch.stack([d.mean for d in filt_hist])[:, ii]
        x_smooth = torch.stack([d.mean for d in smooth_hist])[:, ii]
        std_filt = torch.stack([torch.sqrt(d.cov[ii, ii]) for d in filt_hist])
        std_smooth = torch.stack([torch.sqrt(d.cov[ii, ii]) for d in smooth_hist])

        ax.plot(t, x_true[:, ii], "k", label="true")
        ax.plot(t, x_filt, "C0", label="filter mean")
        ax.fill_between(t, x_filt - 2 * std_filt, x_filt + 2 * std_filt, color="C0", alpha=0.15)
        ax.plot(t, x_smooth, "C1", label="smoother mean")
        ax.fill_between(t, x_smooth - 2 * std_smooth, x_smooth + 2 * std_smooth, color="C1", alpha=0.15)
        ax.set_ylabel(labels[ii])
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time")
    fig.suptitle("Linear-Gaussian damped oscillator: KF vs RTS smoother")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
