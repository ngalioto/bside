import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import argparse

def dxdt(x):
    return np.array([x[1], -x[0]])

argparse = argparse.ArgumentParser()
argparse.add_argument("--x0", type=float, nargs='+', default=[1, 0])
argparse.add_argument("--noise_std", type=float, nargs='+', default=1e-1)
args = argparse.parse_args()

x0 = np.array([1, 0])
print(x0.shape)
print(dxdt(x0).shape)

t_span = (0, 10)
t = np.linspace(*t_span, 100)
x = solve_ivp(lambda t, x: dxdt(x), t_span, y0=x0, t_eval=t)['y'].T
y = x + np.random.normal(0, args.noise_std, x.shape)
plt.plot(x)
plt.plot(y, 'k.')
# plt.show()

