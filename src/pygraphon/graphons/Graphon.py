from abc import ABC, abstractclassmethod
import numpy as np


class Graphon(ABC):
    """Abstract class for Graphon.
    All graphons of this class will be scaled graphon, meaning the integral of f(x,y) over [0,1]^2 is 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.graphon_function = self.graphon_function_builder()
        self.check_graphon()

    @abstractclassmethod
    def graphon_function_builder(self):
        """
        Build the graphon function f(x,y)
        """
        pass

    @abstractclassmethod
    def check_graphon(self):
        """
        Check graphon integrates to 1 and other properties depending on the subclass that iplements it

        Raises:
            ValueError: if graphon does not integrate to 1 it it cannot be automatically scaled
        """
        pass


    def draw(self, rho: float, n: int, exchangeable: bool = True) -> np.ndarray:
        """Draw a graph from the graphon with a given density and number of vertices.

        Args:
            rho (float): edge density of the realized graph
            n (int): number of vertices of the realized graph
            exchangeable (bool, optional): if True, the graph is exchangeable. Defaults to True.

        Returns:
            np.ndarray: adjacency matrix of the realized graph (nxn)
        """
        probs = self._get_edge_probabilities(n, exchangeable=exchangeable, wholeMatrix= True)
        return self._generate_adjacency_matrix(n, probs, rho)

    def _generate_adjacency_matrix(self, n, probs, rho):
        """
        Generate adjacency matrix A_ij = 1 with probability rho*probs_ij

        Parameters
        ----------
        n: int
            number of nodes in the graph (A will be a nxn matrix)
        probs: np.ndarray or int
            bernoulli probability of having an edge if np.ndarray size n*(n-1)/2 with indices corresponding to np.tri_indices(n,1)
            if int : constant probability for all the edges
        rho: float
            edge density of the realized graph

        Returns
        -------
        np.array
            adjacency matrix ind realisations of Bern(probs) (nxn)
        """
        if type(probs) != int:
            assert probs.shape[0] == int(
                n * (n - 1) / 2
            ), f"probs array wrong size compared to number of nodes: got {probs.shape} instead of { n*(n-1)/2}"

        # generate bernoulli draws with help of uniform rv
        r = (np.random.uniform(size=int(n * (n - 1) / 2)) < rho * probs) * 1

        a = np.zeros((n, n))

        # fiil triangles of the matrix
        a[np.triu_indices(n, 1)] = r
        a[np.tril_indices(n, -1)] = a.T[np.tril_indices(n, -1)]
        return a

    def _get_edge_probabilities(
        self,
        n: int,
        exchangeable: bool = True,
        wholeMatrix: bool = True,
    ) -> np.ndarray:
        """Generate a matrix P_ij with  0 =< i,j <= n-1

        Args:
            n (int): number of nodes in the edge probability matrix returned
            exchangeable (bool, optional): if True the graph will be vertex exchangeable. Defaults to True.
            wholeMatrix (bool, optional): if True return the square symmetric matrix, otherwise return the upper diagonal. Defaults to True.

        Returns:
            np.ndarray: matrix of edge probabilities (nxn) if wholeMatrix (n*(n-1)/2) if not wholeMatrix
        """

        latentVarArray = (
            np.random.uniform(0, 1, size=n) if exchangeable else np.array([i / n for i in range(n)])
        )

        # generate edge probabilities from latent variables array
        probs = np.zeros((int(n * (n - 1) / 2), 1)).reshape(-1)

        # TODO: performance
        # Now we iterate over the probabilities array and call the function for each
        # latent variable. Way to do that in a vectorized fashion to avoid for loop ?
        I, J = np.triu_indices(n, 1)
        for index, nodes in enumerate(zip(I, J)):
            probs[index] = self.graphon_function(latentVarArray[nodes[0]], latentVarArray[nodes[1]])

        if wholeMatrix:
            P = np.zeros((n, n))
            P[np.triu_indices(n, 1)] = probs
            P[np.tril_indices(n, -1)] = P.T[np.tril_indices(n, -1)]
            return P
        else:
            return probs
