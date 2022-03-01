from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure

from pygraphon.graphons.GraphonFunction import Graphon
from pygraphon.graphons.StepGraphon import StepGraphon


def plot_graphon_function(
    graphon: Graphon, fig: Figure = None, ax: Axes = None, figsize: Tuple[int, int] = (6, 5)
) -> Tuple[Figure, Axes]:
    """Plot the graphon."""
    x1, x2 = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    y = graphon.graphon_function(x1, x2)
    plt.imshow(y, extent=[0, 1, 0, 1], cmap=plt.cm.get_cmap("jet"), origin="lower")
    plt.colorbar()
    plt.show()

    raise NotImplementedError


def plot(
    graphon: StepGraphon, fig: Figure = None, ax: Axes = None, figsize: Tuple[int, int] = (6, 5)
) -> Tuple[Figure, Axes]:
    """Plot the graphon.

    Args:
        graphon (Graphon): graphon to plot.
        fig (Figure, optional): Defaults to None.
        ax (Axes, optional): Defaults to None.
        figsize (Tuple[int, int], optional):. Defaults to (6, 5).

    Returns:
        Tuple[Figure, Axes]: figure and axis of the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    sc = ax.imshow(graphon.graphon, cmap=plt.cm.get_cmap("jet"), aspect="auto")
    fig.colorbar(sc, ax=ax)
    return fig, ax
