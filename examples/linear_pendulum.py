import torch
import bside
import matplotlib.pyplot as plt
from mcmc_samplers import DelayedRejectionAdaptiveMetropolis, SampleVisualizer

class SSM_Estimate(torch.nn.Module):

    def __init__(
        self,
        model: bside.SSM
    ):
        
        super().__init__()
        self.model = model
        self.filter = bside.KalmanFilter(model=model)

    def loss(
        self,
        data: bside.Data,
        init_dist: bside.FilteringDistribution,
        measure_y0: bool,
        params: torch.Tensor | None = None
    ) -> torch.Tensor:

        return self.filter.nlog_marginal_likelihood(data, init_dist, measure_y0, params)
    
    def optimize(
        self, 
        data: bside.Data, 
        init_dist: bside.FilteringDistribution, 
        measure_y0: bool, 
        lr: float = 0.01, 
        epochs: int = 100
    ) -> None:
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            self.filter.model.update()
            print(self.filter.model.dynamics.mat_x)
            loss = self.loss(data, init_dist, measure_y0)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

dt = 0.1
qc = 0.1
g = 9.81
T = 50
xdim = 2
ydim = 2
r = 0.1
measure_y0 = False
t = torch.linspace(0, T*dt, T+1)

x0 = torch.tensor([1.5, 0.])
A_continuous = torch.tensor([[0., 1.], [-g, 0.]])
A = bside.Matrix(torch.linalg.matrix_exp(A_continuous * dt))
C = bside.Matrix(torch.tensor([[1., 0.], [0., 1.]]))
Q = bside.PSDMatrix(torch.tensor([[qc * dt**3 / 3, qc * dt**2 / 2], [qc * dt**2 / 2, qc * dt]]))
R = bside.PSDMatrix(torch.eye(2) * r**2)

dynamics = bside.LinearModel(A)
measurement = bside.LinearModel(C)
        
observation_model = bside.LinearGaussianModel(
    model = measurement,
    noise_cov = R
)

true_sys = bside.SSM(
    xdim = xdim,
    ydim = ydim,
    dynamics = dynamics,
    observations = observation_model
)

x_true, y = true_sys.measure(x=x0, T=T, keep_y0=measure_y0, return_x=True)
data = bside.Data(y=y, u=None)

# plt.figure()
# plt.plot(t if measure_y0 else t[1:], y, '.')
# plt.title('Observations')
# plt.xlabel('Time')
# plt.ylabel('Position (rad)')
# plt.show()

A_model = bside.Matrix(default=torch.eye(xdim), mask=torch.ones(xdim, xdim, dtype=bool), indices=torch.arange(xdim**2))
learnable_dynamics = bside.LinearGaussianModel(
    model = bside.LinearModel(A_model),
    noise_cov = Q
)

sys_model = bside.SSM(
    xdim = xdim,
    ydim = ydim,
    dynamics = learnable_dynamics,
    observations = observation_model
)

P0 = bside.PSDMatrix(0.01 * torch.eye(xdim))
init_dist = bside.FilteringDistribution(x0, P0)
filter = bside.KalmanFilter(model=sys_model)

with torch.no_grad():
    print(filter.nlog_marginal_likelihood(data, init_dist, measure_y0))

ssm_estimate = SSM_Estimate(sys_model)
# print(ssm_estimate.loss(data, init_dist, measure_y0))
# print(ssm_estimate.loss(data, init_dist, measure_y0, params=A.val.flatten()))
# print(ssm_estimate.loss(data, init_dist, measure_y0))
# ssm_estimate.model.update()
# print(ssm_estimate.loss(data, init_dist, measure_y0))

# ssm_estimate.optimize(data, init_dist, measure_y0, lr=1e-2, epochs=100)
# print(A.val)

# with torch.no_grad():
#     y = sys_model.predict(
#         x = x0,
#         u = None,
#         T = T,
#         return_x = False,
#         keep_y0 = measure_y0
#     )

# dmd_operator = torch.linalg.lstsq(data.y[:-1], data.y[1:]).solution
# dmd_dynamics = bside.LinearModel(bside.Matrix(dmd_operator))
# dmd_sys = bside.SSM(
#     xdim = xdim,
#     ydim = ydim,
#     dynamics = dmd_dynamics,
#     observations = observation_model
# )

# y_dmd = dmd_sys.predict(
#     x = x0,
#     u = None,
#     T = T,
#     return_x = False,
#     keep_y0 = measure_y0
# )

# plt.figure()
# plt.plot(t if measure_y0 else t[1:], y[:,0], label='Bayes')
# plt.plot(t if measure_y0 else t[1:], y_dmd[:,0], 'r', label='DMD')
# plt.plot(t if measure_y0 else t[1:], data.y[:,0], 'k.', label='Data')
# plt.plot(t if measure_y0 else t[1:], x_true[:,0], 'k--', label='True')
# plt.title('Observations')
# plt.xlabel('Time')
# plt.ylabel('Position (rad)')
# plt.show()

target = lambda p: -ssm_estimate.loss(data, init_dist, measure_y0, params=p.flatten())
init_sample = A_model.val.detach().flatten()
init_cov = torch.eye(A_model.params.numel()) * 1e-3

dram = DelayedRejectionAdaptiveMetropolis(
    target = target,
    x0 = init_sample,
    cov = init_cov
)

with torch.no_grad():
    samples, log_probs = dram(int(1e4))

print(f'Sampling acceptance rate: {100*dram.acceptance_ratio:.2f}%')

visualizer = SampleVisualizer(samples)
visualizer.triangular_hist(bins=50)
visualizer.chains()
plt.show()