# -*- coding: utf-8 -*-

"""Test of matrices utils function."""

import numpy as np

from pygraphon.utils.utils_matrix import check_symmetric, permute_matrix


def test_permute_matrix_simple():
    """Check permutation is correct for simple 2x2 matrix."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[4, 3], [2, 1]])
    assert np.allclose(permute_matrix(a, (1, 0)), b)


def test_permute_matrix_complex():
    """Check permutation is correct for 3x3 matrix."""
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    assert np.allclose(permute_matrix(a, (2, 1, 0)), b)


def test_check_symmetric():
    """Check that the symmetric test is correct."""
    a = np.array([[1, 2], [2, 1]])
    assert check_symmetric(a)
