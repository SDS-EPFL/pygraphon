import networkx as nx
import numpy as np

from pygraphon.utils.utils_matrix import check_symmetric


def get_adjacency_matrix_from_graph(graph: nx.Graph) -> np.ndarray:
    """Get the adjacency matrix from a networkx graph

    Args:
        graph (nx.Graph): graph

    Returns:
        np.ndarray: adjacency matrix
    """
    return np.array(nx.adjacency_matrix(graph).todense())


def edge_density(adjacency_matrix: np.ndarray) -> float:
    """Compute the edge density of a sinmple graph represented by an adjacency matrix

    Args:
        adjacency_matrix (np.ndarray): adjacency matrix representing the graph

    Returns:
        float: edge density rho
    """
    return np.sum(adjacency_matrix) / (adjacency_matrix.shape[0] * (adjacency_matrix.shape[1] - 1))


def check_simple_adjacency_matrix(adjacency_matrix: np.ndarray) -> None:
    """Check if an adjacency matrix is symmetric, binary and has no element on the diagonal


    Args:
        adjacency_matrix (np.ndarray): matrix representing the graph to check.

    Raises:
        ValueError: if self loops, non-binary or non-symmetric
    """

    if np.sum(np.diag(adjacency_matrix)) != 0:
        raise ValueError("Adjacency matrix should not contain self-loops")
    if not check_symmetric(adjacency_matrix):
        raise ValueError("Adjacency matrix should be symmetric")
    if not np.all(np.logical_and(adjacency_matrix >= 0, adjacency_matrix <= 1)):
        raise ValueError("Adjacency matrix should be binary")
