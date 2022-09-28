# -*- coding: utf-8 -*-

"""Test of graphons classes."""

import numpy as np
import pytest

from pygraphon.graphons.StepGraphon import StepGraphon


@pytest.fixture(scope="class")
def theta():
    """Create a theta matrix."""
    return np.array([[0.8, 0.2], [0.2, 0.8]])


@pytest.fixture(scope="class")
def block_size(theta):
    """Compute the block size."""
    return 1 / theta.shape[0]


@pytest.fixture(scope="class")
def step_graphon(theta, block_size):
    """Create a step graphon."""
    return StepGraphon(graphon=theta, bandwidthHist=block_size)


class TestRegularGraphon:
    """Test a regular step graphon where all blocks have same size."""

    def test_instantiation(self, step_graphon, theta, block_size):
        """Test that the step graphon is correctly instantiated."""
        # compute theoretical quantities
        areas = np.ones_like(theta) / theta.shape[0] ** 2
        integral = np.sum(theta * areas)

        # check demormalized graphon is equal to theta
        assert np.allclose(step_graphon.graphon * step_graphon.initial_rho, theta)
        # check block size correctly instantiated
        assert step_graphon.bandwidthHist == block_size
        # check areas
        assert np.allclose(step_graphon.areas, areas)
        # check initial rho is correct
        assert step_graphon.initial_rho == integral

    def test_normalized(self, step_graphon):
        """Test that the step graphon is normalized."""
        assert step_graphon.integral() == 1

    def test_edge_probability_between_0_and_1(self, step_graphon):
        """Test that the edge probability is correct."""
        P_ij = step_graphon._get_edge_probabilities(n=20, wholeMatrix=False)
        assert np.all(P_ij <= 1)
        assert np.all(P_ij >= 0)

    def test_edge_probability_correct(self, step_graphon):
        P_small = step_graphon._get_edge_probabilities(n=4, wholeMatrix=True, exchangeable=False)
        P_theoric = np.array(
            [[0, 0.8, 0.2, 0.2], [0.8, 0, 0.2, 0.2], [0.2, 0.2, 0, 0.8], [0.2, 0.2, 0.8, 0]]
        )
        assert np.allclose(P_small, P_theoric)


@pytest.fixture(scope="class")
def theta_irregular():
    """Create a theta matrix."""
    return np.array(
        [[0.8, 0.2, 0.1, 0.01], [0.2, 0.7, 0.2, 0.03], [0.1, 0.2, 0.8, 0.3], [0.01, 0.03, 0.3, 0.9]]
    )


@pytest.fixture(scope="class")
def irregular_step_graphon(theta_irregular):
    """Create a step graphon."""
    return StepGraphon(graphon=theta_irregular, bandwidthHist=0.3)


class TestIrregularGraphon(TestRegularGraphon):
    """Test an irregular step graphon where blocks have different sizes.

    Have to use tolerance because of numerical errors.
    """

    def test_instantiation(self, irregular_step_graphon, theta_irregular):
        """Test that the step graphon is correctly instantiated."""
        # compute theoretical quantities
        areas = np.array(
            [
                [0.09, 0.09, 0.09, 0.03],
                [0.09, 0.09, 0.09, 0.03],
                [0.09, 0.09, 0.09, 0.03],
                [0.03, 0.03, 0.03, 0.01],
            ]
        )
        integral = np.sum(theta_irregular * areas)

        # check demormalized graphon is equal to theta
        assert np.allclose(
            irregular_step_graphon.graphon * irregular_step_graphon.initial_rho, theta_irregular
        )
        # check block size correctly instantiated
        assert irregular_step_graphon.bandwidthHist == 0.3
        # check areas
        assert np.allclose(irregular_step_graphon.areas, areas)
        # check initial rho is correct
        assert np.allclose(irregular_step_graphon.initial_rho, integral, atol=1e-14)

    def test_normalized(self, irregular_step_graphon):
        """Test that the step graphon is normalized."""
        assert np.allclose(irregular_step_graphon.integral(), 1, atol=1e-14)

    def test_edge_probability_between_0_and_1(self, irregular_step_graphon):
        """Test that the edge probability is correct."""
        P_ij = irregular_step_graphon._get_edge_probabilities(n=20, wholeMatrix=False)
        assert np.all(P_ij <= 1)
        assert np.all(P_ij >= 0)
