from collections import Counter
from typing import Tuple

import matlab.engine
import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.utils.utils_graph import edge_density
from pygraphon.utils.utils_maltab import getMatlabPaths, npArray2Matlab, setupMatlabEngine


class HistogramEstimator(BaseEstimator):
    """Implements the histogram estimator from Universality of block model approximation [1] to approximate a graphon from a single adjacency matrix.
        Size of blocks can be  determined automaticaly.

    Args:
        matlabEngine (matlab.engine.MatlabEngine): matlab engine to do the approximation.
        bandwithHist (float, optional): size of the block of the histogram. If None, automatically derived from the observation. Defaults to None.


     Sources:
            [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
    """

    def __init__(
        self, matlabEngine: matlab.engine.MatlabEngine, bandwithHist: float = None
    ) -> None:
        super().__init__()
        self.matlabEngine = setupMatlabEngine(
            matlabEngine=matlabEngine, paths=getMatlabPaths("nethist")
        )
        self.bandwidthHist = bandwithHist

    def approximateGraphonFromAdjacency(self, adjacency_matrix: np.ndarray) -> StepGraphon:
        H, P, h = self._approximate(adjacency_matrix, self.bandwidthHist, self.matlabEngine)
        self.bandwidthHist = h
        return StepGraphon(H, self.bandwidthHist)

    def _approximate(
        adjacencyMatrix: np.ndarray,
        bandwidthHist: float,
        matlabEngine: matlab.engine.matlabengine.MatlabEngine,
    ) -> Tuple[np.ndarray]:
        """Use function from Universality of block model approximation [1] to approximate a graphon from a single adjacency matrix.

        Args:
            adjacencyMatrix (np.ndarray): adjacency matrix of the realized graph
            bandwidthHist (float, optional):  size of the block of the histogram. Defaults to None
            eng (matlab.engine.matlabengine.MatlabEngine, optional): matlab engine to do the approximation. Defaults to None.
            pathToMatlabScripts (str, optional): paths to the matlab scripts for network histogram approximation, used if not matlab engine is given. Defaults to None.

        Raises:
            ValueError: if no matlab engine is given and no path to maltab scripts is given

        Returns:
            Tuple[np.ndarray], float : (H,P),h , H is the block model graphon and P is the edge probability matrix
            corresponding to the adjacency matrix. h is the size of the block


        Sources:
            [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
        """

        # network histogram approximation
        # calls matlab script from paper
        if bandwidthHist is None:
            idx, h = matlabEngine.nethist(npArray2Matlab(adjacencyMatrix), nargout=2)
            bandwidthHist = h / len(idx)
        else:
            # needs this weird conversion for matlab to work, does not accept int
            argh = float(int(bandwidthHist * adjacencyMatrix.shape[0]))

            idx = matlabEngine.nethist(npArray2Matlab(adjacencyMatrix), argh, nargout=1)
        groupmembership = [elt[0] for elt in idx]

        # compute the actual values of the graphon approximation
        groups = np.unique(groupmembership)
        countGroups = Counter(groupmembership)
        ngroups = len(groups)
        rho = edge_density(adjacencyMatrix)
        rho_inv = 1 / rho if rho != 0 else 1
        H = np.zeros((ngroups, ngroups))

        # compute the number of links between groups i and j / all possible links
        for i in range(ngroups):
            for j in np.arange(i, ngroups):
                total = countGroups[groups[i]] * countGroups[groups[j]]
                H[i][j] = (
                    np.sum(
                        adjacencyMatrix[np.where(groupmembership == groups[i])[0]][
                            :, np.where(groupmembership == groups[j])[0]
                        ]
                    )
                    / total
                )
                H[i, j] = H[i, j] * rho_inv
                H[j][i] = H[i, j]

        # fills in the edge probability matrix from the value of the graphon
        P = np.zeros((len(groupmembership), len(groupmembership)))
        for i in range(P.shape[0]):
            for j in np.arange(i + 1, P.shape[0]):
                P[i, j] = H[int(groupmembership[i]) - 1, int(groupmembership[j]) - 1]
                P[j, i] = P[i, j]
        return H, P, bandwidthHist
