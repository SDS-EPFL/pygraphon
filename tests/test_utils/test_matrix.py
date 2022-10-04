import numpy as np

from pygraphon.utils.utils_matrix import permute_matrix


def test_permute_matrix_simple():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[4, 3], [2, 1]])
    print(permute_matrix(a, (1, 0)))
    assert np.allclose(permute_matrix(a, (1, 0)), b)


def test_permute_matrix_complex():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    assert np.allclose(permute_matrix(a, (2, 1, 0)), b)
