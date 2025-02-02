import matplotlib.pyplot as plt
import torch
from bside.filtering import FilteringDistribution
from typing import Iterable, Tuple

def collate_filtering_distributions(
    dists = Iterable[FilteringDistribution]
) -> Tuple[torch.Tensor, torch.tensor]:
    
    means = torch.stack([dist.mean for dist in dists])
    covs = torch.stack([dist.cov for dist in dists])
    
    return means, covs

def plot_filtering_distributions(
    means: torch.Tensor,
    covs: torch.Tensor,
    t: torch.Tensor = None,
    labels: Iterable[str] = None,
    colors: Iterable[str] = None,
    alpha: float = 0.5
) -> plt.Figure:
    
    std_devs = torch.sqrt(covs.diagonal(dim1=-2, dim2=-1))
    t = torch.arange(means.shape[0]) if t is None else t
    
    fig, ax = plt.subplots()
    xdim = means.shape[1]
    for i in range(xdim):
        ax.plot(t, means[:, i], label=labels[i] if labels is not None else None, color=colors[i] if colors is not None else None)
        ax.fill_between(
            t,
            means[:, i] - 2 * std_devs[:, i],
            means[:, i] + 2 * std_devs[:, i],
            alpha=alpha
        )
    ax.legend()
    return fig