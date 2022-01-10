import networkx as nx
import numpy as np


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
    return np.sum(adjacency_matrix) / \
        (adjacency_matrix.shape[0] * (adjacency_matrix.shape[1] - 1))
