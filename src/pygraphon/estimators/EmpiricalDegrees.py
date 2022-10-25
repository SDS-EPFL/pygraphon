"""Implementation of empirical degrees based estimator."""
from typing import Optional, Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon


class LG(BaseEstimator):
    """Estimate graphons based on empirical degrees.

    Parameters
    ----------
    K : int
        Number of blocks.
    """

    def __init__(self, K: int) -> None:
        if K < 1:
            raise ValueError("K must be greater than 0")
        self.K = K

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, K: Optional[int] = None
    ) -> Tuple[Graphon, np.ndarray]:
        if K is None:
            K = self.K
        if K > adjacency_matrix.shape[0]:
            raise ValueError("K must be smaller than the number of nodes")
        raise NotImplementedError()
