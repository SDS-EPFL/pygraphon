import networkx as nx
import numpy as np


def get_ajacency_matrix_from_graph(graph: nx.Graph) -> np.ndarray:
    return np.array(nx.adjacency_matrix(graph).todense())

