"""Implementation of the SBA estimator."""
from typing import Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon


class SBA(BaseEstimator):
    """Estimate graphon base on SB approximation.

    Parameters
    ----------
    delta : float, optional
        precision parameter larger than 0, by default 0.2
    """

    def __init__(self, delta: float = 0.2) -> None:
        if delta < 0:
            raise ValueError("delta must be larger than 0")
        self.delta = delta

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[Graphon, np.ndarray]:
        raise NotImplementedError()
