import numpy as np
from typing import Tuple
from copy import copy


def rearangeMatrix(A, indices):
    return A[indices][:, indices]


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def permute_matrix(matrix: np.ndarray, permutation: Tuple) -> np.ndarray:
    """Permute a matrix according to a permutation.

    Args:
        matrix (np.ndarray): matrix to permute
        permutation (Tuple): permutation to apply: (4,2,1,3)  is interpreted as (1,2,3,4) -> (4,2,1,3)

    Returns:
        np.ndarray: permuted matrix
    """
    new_matrix = copy(matrix)
    new_matrix[permutation, :] = matrix[:, permutation]
    new_matrix[:, permutation] = new_matrix[permutation, :]
    return new_matrix
