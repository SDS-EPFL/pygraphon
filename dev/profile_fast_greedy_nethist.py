import numpy as np

from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.matlab_functions.nethist import nethist


if __name__ == "__main__":
    theta = np.array([[0.6, 0.2, 0.3], [0.2, 0.7, 0.1], [0.3, 0.1, 0.6]])
    graphon = StepGraphon(theta, 1 / 3)
    n = 99
    A = graphon.draw(0.2, n, exchangeable=False)
    argh = n * 1 / 3
    h, estMSqrd, trace = nethist(A, argh, verbose=True, trace=True)
