"""Implementation of neighborhood smoothing estimator."""
from typing import Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon


class NBD(BaseEstimator):
    """Estimate graphon by neighborhood smoothing."""

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[Graphon, np.ndarray]:
        raise NotImplementedError()
