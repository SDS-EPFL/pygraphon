from abc import abstractclassmethod

import networkx as nx
import numpy as np

from pygraphon.graphons.GraphonAbstract import GraphonAbstract
from pygraphon.utils.utils_graph import get_adjacency_matrix_from_graph


class BaseEstimator:
    """Base class for Graphon estimator."""

    def estimate(
        self, graph: nx.Graph = None, adjacency_matrix: np.ndarray = None, *args, **kwargs
    ) -> GraphonAbstract:
        """Estimate the graphon function f(x,y) from a realized graph or adjacency matrix

        Args:
            graph (nx.Graph, optional): networkx simple graph. Defaults to None.
            adjacency_matrix (np.ndarray, optional): adjancency matrix representing the graph. Defaults to None.
        """

        if graph is None and adjacency_matrix is None:
            raise ValueError("graph or adjacency_matrix must be provided")
        if graph is not None and adjacency_matrix is not None:
            if not np.allclose(adjacency_matrix, get_adjacency_matrix_from_graph(graph)):
                raise ValueError("Graph and adjacency_matrix are not consistent")
        if graph is not None:
            adjacency_matrix = get_adjacency_matrix_from_graph(graph)
        return self._approximate_graphon_from_adjacency(adjacency_matrix, *args, **kwargs)

    @abstractclassmethod
    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, *args, **kwargs
    ) -> GraphonAbstract:
        """Estimate the graphon function f(x,y) from an adjacency matrix"""
