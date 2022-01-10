from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
import numpy as np


class MomentEstimator(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def approximateGraphonFromAdjacency(self, adjacency_matrix: np.ndarray) -> StepGraphon:
        raise NotImplementedError()