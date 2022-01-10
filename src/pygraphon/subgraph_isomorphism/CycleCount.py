import matlab.engine
import numpy as np
from pygraphon.utils.utils_maltab import getMatlabPaths, setupMatlabEngine, npArray2Matlab


class CycleCount:
    """Implement the cycle count algorithm for finding the number of cycles in a graph.

    The algorithm is based on the paper:
    """

    def __init__(self, matlab_engine: matlab.engine.MatlabEngine, L: int = 9) -> None:

        assert L >= 3, "input L should be an integer >= 3"
        assert L < 10, "cycleCount algorithm cannot handle L > 10"

        # float conversion needed for matlab... Yeah I know, but ... well ...
        self.L = float(int(L))
        self.matlab_engine = setupMatlabEngine(eng=matlab_engine, paths=getMatlabPaths())

    def __call__(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Count the densities of subgraph C_l in a graph G: t(C_L,G)

        Args:
            adjacency_matrix (np.ndarray): Adjacency matrix representing the simple graph G

        Returns:
            np.ndarray : counts of cycle of length 3 to  L
        """
        t = np.asarray(
            self.matlab_engine.cyclecount(npArray2Matlab(adjacency_matrix), self.L, nargout=1)
        ).flatten()
        return (t ** (np.arange(0, len(t)) + 1))[2:]
