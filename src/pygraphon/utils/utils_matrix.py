from copy import copy
from typing import List, Tuple

import numpy as np


def rearangeMatrix(A, indices):
    """Rearange a matrix according to a permutation."""

    return A[indices][:, indices]


def check_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
    """Check if an array is symmetric.

    Args:
        a ([np.ndarray]): array to check
        rtol ([float], optional): relative tolerance. Defaults to 1e-05.
        atol ([float], optional): absolute tolerance. Defaults to 1e-08.

    Returns:
        [bool]: True if the array is symmetric, False otherwise.
    """
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


def upper_triangle_values(array):
    """Return the upper triangle values of an array.

    Args:
        array ([np.ndarray]): original array

    Returns:
        [np.ndarray]: upper triangle values
    """
    return array[np.triu_indices(array.shape[0])]


def bound_away_from_one_and_zero_arrays(
    arrays: List[np.ndarray], eps: float = np.spacing(1)
) -> List[np.ndarray]:
    """
    This function is used to bound away from 1 and 0 in the log likelihood.
    This is done to avoid numerical issues.
    """
    for array in arrays:
        array[array <= 0] = eps
        array[array >= 1] = 1 - eps
    return arrays
