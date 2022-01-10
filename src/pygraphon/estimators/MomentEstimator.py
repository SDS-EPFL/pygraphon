from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
import numpy as np


class MomentEstimator(BaseEstimator):
    """Moment estimator uses subgraph isomorphims density to fit a step graphon.
    """

    def __init__(self) -> None:
        """initialize the moment estimator.
        """

        super().__init__()

    def approximateGraphonFromAdjacency(self, adjacency_matrix: np.ndarray) -> StepGraphon:
        """Estimate the graphon function f(x,y) from an adjacency matrix"""

        raise NotImplementedError()
