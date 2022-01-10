from ..graphons.Graphon import Graphon
from matplotlib.pyplot import Figure, Axes
from typing import Tuple
import matplotlib.pyplot as plt


def plot(
    graphon: Graphon, fig: Figure = None, ax: Axes = None, figsize: Tuple[int, int] = (6, 5)
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
