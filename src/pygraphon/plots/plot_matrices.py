import matplotlib.pyplot as plt
import numpy as np


def spy(A: np.ndarray, fig=None, ax=None):
    """Spy function for matplotlib.pyplot.

    Args:
        A ([np.ndarray]): matrix to plot.

    Returns:
        fig, ax: figure and axis of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.spy(A, precision=0.01, markersize=0.1)
    return fig, ax
