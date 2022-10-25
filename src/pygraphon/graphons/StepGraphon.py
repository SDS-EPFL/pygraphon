"""Stepgraphon class represent all stepfunction approximation of a continuous graphon."""
import math
from typing import Callable

import numpy as np

from pygraphon.graphons.Graphon import Graphon
from pygraphon.utils.utils_matrix import check_symmetric


class StepGraphon(Graphon):
    """A step function graphon, by giving the matrix representing the block model approxumation.

    Parameters
    ----------
        graphon : np.ndarray
            np array representing the theta matrix
        bandwidthHist : float
            size of the groups (between 0 and 1), by default None
        initial_rho : float
            initial edge density (used to keep track in case of normalization), by default None
    """

    def __init__(
        self, graphon: np.ndarray, bandwidthHist: float, initial_rho: float = None
    ) -> None:
        """Create an instance of a step function graphon, by giving the matrix representing the block model approx.

        Parameters
        ----------
        graphon : np.ndarray
            np array representing the theta matrix
        bandwidthHist : float
            size of the groups (between 0 and 1), by default None
        initial_rho : float
            initial edge density (used to keep track in case of normalization), by default None
        """
        # save args
        self.graphon = graphon
        self.bandwidthHist = bandwidthHist
        assert self.graphon.shape[0] == int(math.ceil(1 / self.bandwidthHist))

        self.areas = np.ones_like(self.graphon) * self.bandwidthHist**2
        self.remainder = 1 - int(1 / self.bandwidthHist) * self.bandwidthHist
        if self.remainder != 0:
            self.areas[:, -1] = self.bandwidthHist * self.remainder
            self.areas[-1, :] = self.bandwidthHist * self.remainder
            self.areas[-1, -1] = self.remainder**2

        super().__init__(function=self.graphon_function_builder(), initial_rho=initial_rho)

    def graphon_function_builder(self) -> Callable:
        """Build the graphon function f(x,y).

        Returns
        -------
        Callable
            graphon function
        """

        def _function(x: float, y: float) -> float:
            """Return the value of the graphon at the point (x,y).

            Parameters
            ----------
            x : float
                coordinate x (first latent variable)
            y : float
                coordinate y (second latent variable)

            Returns
            -------
            float
                f(x,y)
            """
            return self.graphon[int(x // self.bandwidthHist)][int(y // self.bandwidthHist)]

        return _function

    def correct_graphon_integral(self):
        """Normalize the graphon such that the integral is equal to 1 if needed."""
        self.normalize()

    def check_graphon(self):
        """Check if the graphon is symmetric, positive.

        Raises
        ------
        ValueError
             if the graphon is not symmetric
        ValueError
            if the graphon is not non negative
        """
        if not check_symmetric(self.graphon):
            raise ValueError("graphon matrix should be symmetric")
        if not np.all(self.graphon >= 0):
            raise ValueError("graphon matrix should be non-negative")

    def integral(self, graphon=None, areas=None) -> float:
        """Integrate the graphon over [0,1]x[0,1].

        Parameters
        ----------
        graphon : np.ndarray, optional
            theta matrix, by default None
        areas : np.ndarray, optional
            areas of the different blocks, by default None

        Returns
        -------
        float
            the value of the integral
        """
        if graphon is None:
            graphon = self.graphon
        if areas is None:
            areas = self.areas
        return np.sum(graphon * areas)

    def normalize(self) -> None:
        """Normalize graphon such that the integral is equal to 1.

        If the graphon is the empty graphon, does not do anything
        """
        integral = self.integral()
        if integral != 0:
            self.graphon = self.graphon / self.integral()
        else:
            self.graphon = self.graphon

    def get_graphon(self) -> np.ndarray:
        r"""Get the graphon matrix.

        Returns
        -------
        np.ndarray
            graphon connectivity matrix (\Theta)
        """
        return self.graphon

    def get_number_groups(self) -> int:
        """Return the number of groups of the graphon.

        Returns
        -------
        int
            number of groups
        """
        return int(1 // self.bandwidthHist) + 1
