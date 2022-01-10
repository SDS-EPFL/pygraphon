"""Network histogram class."""
from collections import Counter
from typing import Tuple

import matlab.engine
import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.utils.utils_graph import edge_density
from pygraphon.utils.utils_maltab import getMatlabPaths, npArray2Matlab, setupMatlabEngine


class HistogramEstimator(BaseEstimator):
    """Implements the histogram estimator from Universality of block model approximation [1].
    Approximate a graphon from a single adjacency matrix. Size of blocks can be  determined automaticaly.

    Args:
        matlab_engine (matlab.engine.MatlabEngine): matlab engine to do the approximation.
        bandwithHist (float, optional): size of the block of the histogram. If None, automatically derived
        from the observation. Defaults to None.


     Sources:
            [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
    """

    def __init__(
        self, matlab_engine: matlab.engine.MatlabEngine, bandwithHist: float = None
    ) -> None:
        """Initialize the histogram estimator.

        Args:
             matlab_engine (matlab.engine.MatlabEngine): matlab engine to do the approximation.
        bandwithHist (float, optional): size of the block of the histogram. If None, automatically derived
        from the observation. Defaults to None.
        """

        super().__init__()
        self.matlab_engine = setupMatlabEngine(eng=matlab_engine, paths=getMatlabPaths())
        self.bandwidthHist = bandwithHist

    def _approximateGraphonFromAdjacency(self, adjacency_matrix: np.ndarray) -> StepGraphon:
        """Estimate the graphon function f(x,y) from an adjacency matrix"""

        graphon_matrix, _, h = self._approximate(
            adjacency_matrix, self.bandwidthHist, self.matlab_engine
        )
        self.bandwidthHist = h
        return StepGraphon(graphon_matrix, self.bandwidthHist)

    def _approximate(
        self,
        adjacencyMatrix: np.ndarray,
        bandwidthHist: float,
        matlab_engine: matlab.engine.matlabengine.MatlabEngine,
    ) -> Tuple[np.ndarray]:
        """Use function from Universality of block model approximation [1] to approximate a graphon
        from a single adjacency matrix.

        Args:
            adjacencyMatrix (np.ndarray): adjacency matrix of the realized graph
            bandwidthHist (float, optional):  size of the block of the histogram. Defaults to None
            eng (matlab.engine.matlabengine.MatlabEngine, optional): matlab engine to do the approximation.
            Defaults to None.
            pathToMatlabScripts (str, optional): paths to the matlab scripts for network histogram approximation,
                        used if not matlab engine is given. Defaults to None.

        Raises:
            ValueError: if no matlab engine is given and no path to maltab scripts is given

        Returns:
            Tuple[np.ndarray], float : graphon_matrix, edge_probability_matrix, h. graphon_matrix is the block model
            graphon and P is the edge probability matrix
            corresponding to the adjacency matrix. h is the size of the block


        Sources:
            [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
        """

        # network histogram approximation
        # calls matlab script from paper
        if bandwidthHist is None:
            idx, h = matlab_engine.nethist(npArray2Matlab(adjacencyMatrix), nargout=2)
            bandwidthHist = h / len(idx)
        else:
            # needs this weird conversion for matlab to work, does not accept int
            argh = float(int(bandwidthHist * adjacencyMatrix.shape[0]))

            idx = matlab_engine.nethist(npArray2Matlab(adjacencyMatrix), argh, nargout=1)
        groupmembership = [elt[0] for elt in idx]

        # compute the actual values of the graphon approximation
        groups = np.unique(groupmembership)
        countGroups = Counter(groupmembership)
        ngroups = len(groups)
        rho = edge_density(adjacencyMatrix)
        rho_inv = 1 / rho if rho != 0 else 1
        graphon_matrix = np.zeros((ngroups, ngroups))

        # compute the number of links between groups i and j / all possible links
        for i in range(ngroups):
            for j in np.arange(i, ngroups):
                total = countGroups[groups[i]] * countGroups[groups[j]]
                graphon_matrix[i][j] = (
                    np.sum(
                        adjacencyMatrix[np.where(groupmembership == groups[i])[0]][
                            :, np.where(groupmembership == groups[j])[0]
                        ]
                    )
                    / total
                )
                graphon_matrix[i, j] = graphon_matrix[i, j] * rho_inv
                graphon_matrix[j][i] = graphon_matrix[i, j]

        # fills in the edge probability matrix from the value of the graphon
        edge_probability_matrix = np.zeros((len(groupmembership), len(groupmembership)))
        for i in range(edge_probability_matrix.shape[0]):
            for j in np.arange(i + 1, edge_probability_matrix.shape[0]):
                edge_probability_matrix[i, j] = graphon_matrix[
                    int(groupmembership[i]) - 1, int(groupmembership[j]) - 1
                ]
                edge_probability_matrix[j, i] = edge_probability_matrix[i, j]
        return graphon_matrix, edge_probability_matrix, bandwidthHist
