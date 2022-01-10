from abc import ABC, abstractclassmethod

import networkx as nx
import numpy as np

from pygraphon.graphons.Graphon import Graphon
from pygraphon.utils.utils_graph import get_adjacency_matrix_from_graph


class BaseEstimator(ABC):
    """Base class for Graphon estimator."""

    def __init__(self) -> None:
        """Constructor."""

        super().__init__()

    def estimate(self, graph: nx.Graph = None, adjacency_matrix: np.ndarray = None) -> Graphon:
        """Estimate the graphon function f(x,y) from a realized graph or adjacency matrix

        Args:
            graph (nx.Graph, optional): networkx simple graph. Defaults to None.
            adjacency_matrix (np.ndarray, optional): adjancency matrix representing the graph. Defaults to None.
        """

        if graph is None and adjacency_matrix is None:
            raise ValueError("graph or adjacency_matrix must be provided")
        if graph is not None and adjacency_matrix is not None:
            raise ValueError("graph or adjacency_matrix must be provided, not both")
        if graph is not None:
            adjacency_matrix = get_adjacency_matrix_from_graph(graph)
        return self._approximateGraphonFromAdjacency(adjacency_matrix)

    @abstractclassmethod
    def _approximateGraphonFromAdjacency(self, adjacency_matrix: np.ndarray) -> Graphon:
        """Estimate the graphon function f(x,y) from an adjacency matrix"""
        pass
