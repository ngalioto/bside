<img alt="B-Side logo" src=".github/assets/bside_light.png#gh-light-mode-only">
<img alt="B-Side logo" src=".github/assets/bside_dark.png#gh-dark-mode-only">

# B-Side

A Bayesian system identification toolbox. 

This is an ongoing work in progress to which I dedicate a portion of my personal time.

## Installation

Install from PyPi
```bash
pip install bside
```

Alternatively, install from source to get the latest version
```bash
pip install git+https://github.com/ngalioto/bside.git
```

## Examples

Each example is a notebook with a matching script in [`examples/`](examples):

| | Linear | Nonlinear |
|---|---|---|
| **Filtering & smoothing** | [`linear_filtering`](examples/linear_filtering.ipynb): Kalman, square-root Kalman, RTS, batched | [`nonlinear_filtering`](examples/nonlinear_filtering.ipynb): unscented, Gauss-Hermite, cubature, ensemble, particle; unscented RTS and particle smoothers |
| **System identification** | [`linear_pendulum`](examples/linear_pendulum.ipynb): EM and Bayesian inference (DRAM) | [`lorenz63`](examples/lorenz63.ipynb): UKF and particle-filter likelihoods, Bayesian inference (DRAM) |

## Citation
If you find any portion of this repository useful for your own work, please consider citing this [paper](https://link.springer.com/article/10.1007/s11071-020-05925-8):
```bibtex
@article{galioto2020bayesian,
  title={Bayesian system {ID}: optimal management of parameter, model, and measurement uncertainty},
  author={Galioto, Nicholas and Gorodetsky, Alex Arkady},
  journal={Nonlinear Dynamics},
  volume={102},
  number={1},
  pages={241--267},
  year={2020},
  publisher={Springer}
}
```

## Author information

Author: Nicholas Galioto  
Email: [ngalioto@umich.edu](mailto:ngalioto@umich.edu)  
License: MIT  
Copyright 2025, Nicholas Galioto
