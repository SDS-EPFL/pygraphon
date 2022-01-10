import matplotlib.pyplot as plt


def spy(A, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.spy(A, precision=0.01, markersize=0.1)
    return fig, ax
