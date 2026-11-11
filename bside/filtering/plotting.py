from typing import TYPE_CHECKING, Iterable, Tuple

import torch

from bside.filtering import FilteringDistribution

if TYPE_CHECKING:
    from matplotlib.figure import Figure

def _moments(
    dist: FilteringDistribution
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance of a Gaussian, weighted-particle or ensemble distribution."""

    if dist.mean is not None and dist.cov is not None:
        return dist.mean, dist.cov
    if dist.log_weights is not None:
        mean = dist.weighted_mean()
        return mean, dist.weighted_cov(mean)
    mean = dist.particles.mean(-2)
    res = dist.particles - mean.unsqueeze(-2)
    return mean, res.mT @ res / (dist.size - 1)


def collate_filtering_distributions(
    dists: Iterable[FilteringDistribution]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack a filtering / smoothing history into ``(means, covs)`` tensors."""

    means, covs = zip(*(_moments(dist) for dist in dists))
    return torch.stack(means), torch.stack(covs)


def plot_filtering_distributions(
    means: torch.Tensor,
    covs: torch.Tensor,
    t: torch.Tensor = None,
    labels: Iterable[str] = None,
    colors: Iterable[str] = None,
    alpha: float = 0.5
) -> "Figure":

    # matplotlib is optional (``pip install bside[plot]``); import on use only.
    import matplotlib.pyplot as plt

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