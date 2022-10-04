# -*- coding: utf-8 -*-

"""Test of graphons classes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygraphon.graphons.GraphonFunction import Graphon
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.norm.MSE_graphon import permutation_distance


@pytest.fixture
def graphon():
    """Create a step graphon"""
    return StepGraphon(
        graphon=np.array(
            [
                [0.8, 0.3, 0.2],
                [0.3, 0.5, 0.34],
                [0.2, 0.34, 0.7],
            ]
        ),
        bandwidthHist=1 / 3,
    )


def test_cannot_use_mse_not_step_graphon():
    graphon = Graphon(function=lambda x, y: 0.5)
    with pytest.raises(TypeError):
        permutation_distance(graphon, graphon)


def test_cannot_compare_graphon_not_same_bandwith(graphon):
    graphon_diff = StepGraphon(graphon=np.array([[0.8, 0.5], [0.5, 0.9]]), bandwidthHist=1 / 2)
    with pytest.raises(NotImplementedError):
        permutation_distance(graphon, graphon_diff)


def test_cannot_compare_graphon_not_mse_nor_mae(graphon):
    with pytest.raises(ValueError):
        permutation_distance(graphon, graphon, norm="not_implemented")


@pytest.mark.parametrize(
    "norm",
    ["MSE", "MAE"],
)
def test_mse_values(graphon, norm):

    # define permuted version of first graphon
    graphon2 = StepGraphon(
        graphon=np.array([[0.8, 0.2, 0.3], [0.2, 0.7, 0.34], [0.3, 0.34, 0.5]]), bandwidthHist=1 / 3
    )
    assert_allclose(permutation_distance(graphon, graphon, norm=norm), 0, atol=1e-15)
    assert_allclose(permutation_distance(graphon, graphon2, norm=norm), 0, atol=1e-15)
