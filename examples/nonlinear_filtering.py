import torch
from bside.filtering import UnscentedKalmanFilter, FilteringDistribution
from bside.filtering.plotting import *
from bside.models import PSDMatrix
from bside.dynamics import NonlinearAdditiveModel
from bside.state_space import SSM
from bside.dataset import Data
import matplotlib.pyplot as plt

dt = 0.01
qc = 0.1
g = 9.81
xdim = 2
ydim = 1
T = 500
r = 0.1
measure_y0 = False
t = torch.linspace(0, T*dt, T+1)

x0 = torch.tensor([1.5, 0.])
Q = PSDMatrix(torch.tensor([[qc * dt**3 / 3, qc * dt**2 / 2], [qc * dt**2 / 2, qc * dt]]))
R = PSDMatrix(torch.tensor([[r**2]]))

def dynamics(x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    return torch.cat([x[:, 0:1] + x[:, 1:2] * dt, x[:, 1:2] - g * torch.sin(x[:, 0:1]) * dt], dim=1)
def measurement(x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    return torch.sin(x[:, 0:1])



dynamics_model = NonlinearAdditiveModel(
    f = dynamics,
    noise_cov = Q,
    in_dim = xdim,
    out_dim = xdim
)
        
observation_model = NonlinearAdditiveModel(
    f = measurement,
    noise_cov = R,
    in_dim = xdim,
    out_dim = ydim
)

sys = SSM(
    xdim = xdim,
    ydim = ydim,
    dynamics = dynamics_model,
    observations = observation_model
)

x_true, y = sys.measure(x=x0, T=T, keep_y0=measure_y0, return_x=True)
data = Data(y=y, u=None)

P0 = PSDMatrix(0.1*torch.eye(xdim))
init_dist = FilteringDistribution(x0, P0)
filter = UnscentedKalmanFilter(model=sys)

xf, lp = filter.filter(
    data=data, 
    init_dist=init_dist, 
    y0=measure_y0, 
    return_history=True,
    compute_log_prob=True
)
print(f'Log probability: {lp.item():.4f}')
m_filtered, P_filtered = collate_filtering_distributions(xf)

plot_filtering_distributions(m_filtered, P_filtered, t, labels=['Position', 'Velocity'])
plt.plot(t[1:], x_true[:, 0], 'k--', label='True Position')
plt.plot(t[1:], x_true[:, 1], 'k--', label='True Velocity')
plt.legend()
plt.show()