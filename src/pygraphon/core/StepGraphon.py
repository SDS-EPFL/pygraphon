import inspect
from collections import Counter
from typing import Callable, Tuple

import matlab
import matlab.engine
import networkx as nx
import numpy as np

from ..utils.utils_graph import get_ajacency_matrix_from_graph
from ..utils.utils_maltab import npArray2Matlab, setupMatlabEngine
from ..utils.utils_maths import generate_all_permutations
from ..utils.utils_matrix import check_symmetric, permute_matrix


def generate_adjacency_matrix(n, probs, rho):
    """
    probs needs to be an array of probability of size n*(n-1)/2, with indices corresponding to np.tri_indices(n,1)
    Parameters
    ----------
    n: int
        number of nodes in the graph (A will be a nxn matrix)
    probs: np.array or int
        bernoulli probability of having an edge
    Returns
    -------
    np.array
        adjacency matrix ind realisations of Bern(probs)
    """
    if type(probs) != int:
        assert probs.shape[0] == int(
            n * (n - 1) / 2
        ), f"probs array wrong size compared to number of nodes: got {probs.shape} instead of { n*(n-1)/2}"

    # generate bernoulli draws with help of uniform rv
    _R = (np.random.uniform(size=int(n * (n - 1) / 2)) < rho * probs) * 1

    P = np.zeros((n, n))

    # fiil triangles of the matrix
    P[np.triu_indices(n, 1)] = _R
    P[np.tril_indices(n, -1)] = P.T[np.tril_indices(n, -1)]
    return P


def norm_graphon_labeled(graphon1, graphon2, norm="MISE"):
    if norm == "MISE":
        result = (np.sum((graphon1 - graphon2) ** 2)) / (graphon2.shape[0] ** 2)
    elif norm == "ABS" or norm == "MAE":
        result = (np.sum(np.abs(graphon1 - graphon2))) / (graphon2.shape[0] ** 2)
    else:
        raise ValueError("norm not defined")
    return result


class StepGraphon:
    def __init__(
        self,
        graph: nx.Graph = None,
        graphon: np.ndarray = None,
        numberNodes: int = None,
        bandwidthHist: float = None,
        matlabEngine: matlab.engine.MatlabEngine = None,
        normalized: bool = True,
    ) -> None:
        """Create an instance of a step function graphon, by either approximating one from a graph,
        or by directly giving the matrix representing the block model approxumation.

        Args:
            graph (nx.Graph, optional): [description]. Defaults to None.
            graphon (np.ndarray, optional): [description]. Defaults to None.
            numberNodes (int, optional): [description]. Defaults to None.
            matlabEngine (matlab.engine.matlabengine.MatlabEngine, optional): [description]. Defaults to None.
            normalized (bool, optional): normalized graphon estimation by inverse of edge density. Defaults to True. (only used if graph is given as an argument and not graphon)

        Raises:
            ValueError: If neither a graph or a graphon is given
        """

        # save args
        self.numberNodes = numberNodes
        self.graphon = graphon
        self.bandwidthHist = bandwidthHist
        self.EdgeProbabilities = None

        # check that there are enough arguments:
        if graph is None and graphon is None:
            raise ValueError(
                "Must either provide the value of the graphon or a graph to perform an approximation"
            )

        # approximate the value of the graphon using blockmodel approx
        if graph is not None:
            if graphon is not None or numberNodes is not None:
                print(
                    "graph argument has precedence over other args, approximating graphon from adjacency matrix..."
                )

            # THIS WILL FAIL IN ANOTHER CODE ARCHITECTURE !
            paths = self.getMatlabPaths()

            # approximate the graphon using the method from Olhede and Wolfe (2013)
            adjacencyMatrix = get_ajacency_matrix_from_graph(graph)
            self.matlabEngine = setupMatlabEngine(matlabEngine, paths)
            (
                self.graphon,
                self.EdgeProbabilities,
                self.bandwidthHist,
            ) = self.approximateGraphonFromAdjacency(
                adjacencyMatrix=adjacencyMatrix,
                bandwidthHist=self.bandwidthHist,
                matlabEngine=self.matlabEngine,
                normalized=normalized,
            )
            self.numberNodes = len(graph.nodes)

        if graphon is not None:
            if check_symmetric(graphon):
                self.graphon = graphon
            else:
                raise ValueError("graphon matrix should be symmetric")

        self.areas = np.ones_like(self.graphon) * self.bandwidthHist ** 2
        self.remainder = 1 - int(1 / self.bandwidthHist) * self.bandwidthHist
        if self.remainder != 0:
            self.areas[:, -1] = self.bandwidthHist * self.remainder
            self.areas[-1, :] = self.bandwidthHist * self.remainder
            self.areas[-1, -1] = self.remainder ** 2

        self.graphon_function = self.graphon_function_builder()

    def graphon_function_builder(self) -> Callable:
        def function(x, y, h=self.bandwidthHist, blocksValue=self.graphon):
            return blocksValue[int(x // h)][int(y // h)]

        return function

    def draw(self, rho: float, numberNodes: int = None, exchangeable: bool = False) -> np.ndarray:

        # check parameters
        nodes = numberNodes if numberNodes is not None else self.numberNodes
        if nodes is None:
            raise ValueError(
                "Must provide number of nodes since no default node number were provided."
            )

        # generate probabilities for the edges
        probs = self.get_edge_probabilities(
            numberNodes=numberNodes, exchangeable=exchangeable, wholeMatrix=False
        )
        A = generate_adjacency_matrix(nodes, probs, rho)

        return A

    @staticmethod
    def approximateGraphonFromAdjacency(
        adjacencyMatrix: np.ndarray,
        bandwidthHist: float = None,
        matlabEngine: matlab.engine.matlabengine.MatlabEngine = None,
        pathToMatlabScripts: str = None,
        normalized: bool = True,
    ) -> Tuple[np.ndarray]:
        """Use function from Universality of block model approximation [1] to approximate a graphon from a single adjacency matrix.
            Size of blocks are determined automaticaly for now.

        Args:
            adjacencyMatrix (np.ndarray): adjacency matrix of the realized graph
            bandwidthHist (float, optional):  size of the block of the histogram. Defaults to None
            eng (matlab.engine.matlabengine.MatlabEngine, optional): matlab engine to do the approximation. Defaults to None.
            pathToMatlabScripts (str, optional): paths to the matlab scripts for network histogram approximation, used if not matlab engine is given. Defaults to None.
            normalized (bool, optional): normalized graphon estimation by inverse of edge density. Defaults to True.

        Raises:
            ValueError: if no matlab engine is given and no path to maltab scripts is given

        Returns:
            Tuple[np.ndarray], float : (H,P),h , H is the block model graphon and P is the edge probability matrix
            corresponding to the adjacency matrix. h is the size of the block


        Sources:
            [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
        """

        # setup for matlab engine
        matlabEngine = setupMatlabEngine(matlabEngine, pathToMatlabScripts)

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
        rho = nx.density(nx.from_numpy_array(adjacencyMatrix))
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
                H[i, j] = H[i, j] * rho_inv if normalized else H[i, j]
                H[j][i] = H[i, j]

        # fills in the edge probability matrix from the value of the graphon
        P = np.zeros((len(groupmembership), len(groupmembership)))
        for i in range(P.shape[0]):
            for j in np.arange(i + 1, P.shape[0]):
                P[i, j] = H[int(groupmembership[i]) - 1, int(groupmembership[j]) - 1]
                P[j, i] = P[i, j]
        return H, P, bandwidthHist

    def getMatlabPaths(self):
        """Dirty trick to get the correct paths of the matlab scripts:
        ### Any change in name or structure of the code directory will make this fail !

        Returns:
            [str]: paths to matlab file for estimating graphon
        """
        pathFile = inspect.getfile(self.__class__)
        path = pathFile.replace("networks/graphon.py", "")
        path += "network_histogram/"
        return path

    def distance(self, stepGraphon, norm: str = "MISE") -> float:
        """Compute the distance between two step fuction graphons

        Args:
            stepGraphon ([StepGraphon]): [description]
            norm (str, optional): [description]. Defaults to "MISE".

        Raises:
            NotImplementedError: [description]
            ValueError: [description]

        Returns:
            float: [description]
        """

        # get the data we need
        graphon1 = self.graphon
        h1 = self.bandwidthHist
        graphon2 = stepGraphon.graphon
        h2 = stepGraphon.bandwidthHist

        # check that graphons have the same bandwidth for their blocks
        if h1 != h2:
            raise NotImplementedError("different size of graphons cannot be compared for now")

        # weight the difference of the graphons based on the area of the block considered
        # compute the weighted norm
        indices_x, indices_y = np.triu_indices(self.graphon.shape[0])
        indices = [(x, indices_y[i]) for i, x in enumerate(indices_x)]

        # check if the graphon blocks all have same size
        if (
            len(np.unique(self.graphon.shape)) != 1
            or len(np.unique(stepGraphon.graphon.shape)) != 1
        ):
            raise ValueError("Cannot compare graphons with heterogeneous block sizes")

        # generate all possible permutations
        permutations_possible = generate_all_permutations(graphon1.shape[0])

        # should be upper bound on the distance possible between two graphons
        norm_value = (np.sum(graphon1) + np.sum(graphon2)) ** 2
        for sigma in permutations_possible:
            if norm == "MISE":
                result = np.sqrt(
                    np.sum(((graphon1 - permute_matrix(graphon2, sigma)) ** 2) * self.areas)
                )
            elif norm in ["ABS", "MAE"]:
                result = np.average(
                    np.sum(np.abs(graphon1 - permute_matrix(graphon2, sigma)) * self.areas)
                )
            else:
                raise ValueError(f"norm not defined, got {norm}")
            norm_value = min(norm_value, result)

        return norm_value

    def norm(self, norm: str = "MISE"):
        empty = StepGraphon(graphon=np.zeros_like(self.graphon), bandwidthHist=self.bandwidthHist)
        return self.distance(empty, norm)

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
        self.graphon = self.get_normalized_graphon()

    def get_graphon(self, normalized=False) -> np.ndarray:
        if normalized:
            return self.get_normalized_graphon()
        else:
            return self.graphon

    def get_normalized_graphon(self) -> np.ndarray:
        integral = self.integral()
        if integral != 0:
            return self.graphon / self.integral()
        else:
            return self.graphon

    def get_number_groups(self) -> int:
        return 1 // self.bandwidthHist + 1

    def get_edge_probabilities(
        self,
        numberNodes: int = None,
        latentVarArray: np.ndarray = None,
        exchangeable: bool = False,
        wholeMatrix: bool = True,
    ) -> np.ndarray:
        """Generate a matrix P_ij with  0 =< i,j <= numberNodes-1 if numberNodes
        is given, otherwise use the number of nodes by default of the grpahon if it was given in the initialization

        Args:
            numberNodes ([int], optional): Number of nodes in the edge probability matrix returned. Defaults to None.

        Raises:
            ValueError: If no numberNodes is given and graphon was not built with either a graph, or a default number of nodes,
            impossible to determine the size of the desired matrix in output.

        Returns:
            np.ndarray: matrix of edge probabilities of size numberNodes x numberNodes
        """
        if latentVarArray is not None:
            latentVarArray = np.squeeze(latentVarArray)
            if latentVarArray.ndim != 1:
                raise ValueError("Latent variables array should be one dimensional")
            if numberNodes is not None and numberNodes != latentVarArray.shape[0]:
                raise ValueError(
                    f"Number of nodes ({numberNodes}) and length of latent variable array ({latentVarArray.shape[0]}) disagree"
                )
            elif numberNodes is None:
                numberNodes = latentVarArray.shape[0]

        elif numberNodes is not None or (
            self.EdgeProbabilities is None and self.numberNodes is not None
        ):
            if numberNodes is None:
                numberNodes = self.numberNodes

            latentVarArray = (
                np.random.uniform(0, 1, size=numberNodes)
                if exchangeable
                else np.array([i / numberNodes for i in range(numberNodes)])
            )

        elif self.EdgeProbabilities is not None:
            return self.EdgeProbabilities

        else:
            raise ValueError(
                "No edge probabilites were computed in the initialization, no default number of ndoes and  no number of nodes were given: unable to return a matrix of edge probabilites"
            )
        # deals with generating edge probabilities from latent variables array
        probs = np.zeros((int(numberNodes * (numberNodes - 1) / 2), 1)).reshape(-1)

        # TODO: performance
        # Now we iterate over the probabilities array and call the function for each
        # latent variable. Way to do that in a vectorized fashion to avoid for loop ?
        I, J = np.triu_indices(numberNodes, 1)
        for index, nodes in enumerate(zip(I, J)):
            probs[index] = self.graphon_function(latentVarArray[nodes[0]], latentVarArray[nodes[1]])

        if wholeMatrix:
            P = np.zeros((numberNodes, numberNodes))
            P[np.triu_indices(numberNodes, 1)] = probs
            P[np.tril_indices(numberNodes, -1)] = P.T[np.tril_indices(numberNodes, -1)]
            return P
        else:
            return probs
