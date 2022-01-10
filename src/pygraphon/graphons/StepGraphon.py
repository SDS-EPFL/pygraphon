from typing import Callable

import numpy as np

from pygraphon.graphons.Graphon import Graphon
from pygraphon.utils.utils_matrix import check_symmetric


class StepGraphon(Graphon):
    """A step function graphon, by giving the matrix representing the block model approxumation.

        Args:
            graphon (np.ndarray): [description]. Defaults to None.
            bandwidthHist (float, optional): [description]. Defaults to None.
    """
    def __init__(
        self,
        graphon: np.ndarray,
        bandwidthHist: float = None,
    ) -> None:
        """Create an instance of a step function graphon, by giving the matrix representing the block model approxumation.

        Args:
            graphon (np.ndarray): [description]. Defaults to None.
            bandwidthHist (float, optional): [description]. Defaults to None.
        """
        super().__init__()

        # save args
        self.graphon = graphon
        self.bandwidthHist = bandwidthHist

        self.areas = np.ones_like(self.graphon) * self.bandwidthHist ** 2
        self.remainder = 1 - int(1 / self.bandwidthHist) * self.bandwidthHist
        if self.remainder != 0:
            self.areas[:, -1] = self.bandwidthHist * self.remainder
            self.areas[-1, :] = self.bandwidthHist * self.remainder
            self.areas[-1, -1] = self.remainder ** 2

        self.graphon_function = self.graphon_function_builder()

    def graphon_function_builder(self) -> Callable:
        """
        Build the graphon function f(x,y)
        """

        def function(x: float, y: float , h: float = self.bandwidthHist, blocksValue: np.ndarray = self.graphon):
            """return the value of the graphon at the point (x,y)

            Args:
                x ([float]):coordinate x
                y ([type]):coordinate y
                h ([float], optional): size of the blocks of the graphon. Defaults to self.bandwidthHist.
                blocksValue ([np.ndarray], optional): connection matrices value. Defaults to self.graphon.

            Returns:
                [float]: f(x,h)
            """
            return blocksValue[int(x // h)][int(y // h)]

        return function

    def check_graphon(self):
        """ check if the graphon is symmetric, positive and normalized

        Raises:
            ValueError: if the graphon is not symmetric,
            ValueError: if the graphon is not non negative
        """
        if not self.integral() == 1:
            Warning.warn("Graphon is not normalized, rescaling ...")
            self.normalize()
        if not check_symmetric(self.graphon):
            raise ValueError("graphon matrix should be symmetric")
        if not np.all(self.graphon >= 0):
            raise ValueError("graphon matrix should be non-negative")

    def integral(self) -> float:
        """Integrate the graphon over [0,1]x[0,1]

        Returns:
            float: the value of the integral
        """
        return np.sum(self.graphon * self.areas)

    def normalize(self) -> None:
        """Normalize graphon such that the integral is equal to 1
        if the graphon is the empty graphon, does not do anything
        """
        integral = self.integral()
        if integral != 0:
            self.graphon = self.graphon / self.integral()
        else:
            self.graphon = self.graphon

    def get_graphon(self) -> np.ndarray:
        """Get the graphon matrix"""
        return self.graphon

    def get_number_groups(self) -> int:
        """Return the number of groups of the graphon
        """
        return 1 // self.bandwidthHist + 1
