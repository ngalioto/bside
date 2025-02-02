import torch
# most of this should just come from bside
from bside.filtering import KalmanFilter, FilteringDistribution
from bside.filtering.plotting import *
from bside.dynamics import LinearModel, LinearGaussianModel
from bside.models import Matrix, PSDMatrix
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
A_continuous = torch.tensor([[0, 1], [-g, 0]])
A = Matrix(torch.linalg.matrix_exp(A_continuous * dt))
C = Matrix(torch.tensor([[1., 0.]]))
Q = PSDMatrix(torch.tensor([[qc * dt**3 / 3, qc * dt**2 / 2], [qc * dt**2 / 2, qc * dt]]))
R = PSDMatrix(torch.tensor([[r**2]]))

dynamics = LinearModel(A)
measurement = LinearModel(C)

dynamics_model = LinearGaussianModel(
    model = dynamics,
    noise_cov = Q    
)
        
observation_model = LinearGaussianModel(
    model = measurement,
    noise_cov = R
)

sys = SSM(
    xdim = xdim,
    ydim = ydim,
    dynamics = dynamics_model,
    observations = observation_model
)

x_true, y = sys.measure(x=x0, T=T, keep_y0=measure_y0, return_x=True)
data = Data(y=y, u=None)

plt.figure()
plt.plot(t if measure_y0 else t[1:], y, '.')
plt.title('Observations')
plt.xlabel('Time')
plt.ylabel('Position (rad)')
plt.show()

P0 = PSDMatrix(torch.eye(xdim))
init_dist = FilteringDistribution(x0, P0)
filter = KalmanFilter(model=sys)

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