import numpy as np

from pygraphon.utils.utils_func import copy_func


def test_different_functions():
    def f(x, y):
        return x + y

    assert copy_func(f) != f


def test_same_output_values():
    def f(x):
        return x**2

    g = copy_func(f)
    assert np.all([f(x) == g(x) for x in range(10)])
