"""Implementation of network histogram estimator."""
from typing import Optional, Tuple

import numpy as np
import scipy
import scipy.sparse.linalg
from loguru import logger
from numba import jit
from scipy import stats

from .assignment import Assignment
from .greedy_readable import greedy_opt


def nethist(
    A: np.ndarray,
    h: Optional[int] = None,
    absTol: float = 2.5 * 1e-4,
    maxNumIterations: int = 500,
    past_non_improving: int = 3,
    progress_bar: bool = True,
    laplacian: bool = False,
) -> Tuple[Assignment, int]:
    """Compute the network histogram of an N-by-N adjacency matrix.

    Adjacency matrix is assumed to be  0-1 valued, symmetric, and with zero on the diagonal.

    Parameters
    ----------
    A : np.ndarray
        adjacency matrix
    h : int
        specifies the number of nodes in each histogram bin,, by default is optimized based on input.
    absTol : float
        absolute tolerance for convergence
    maxNumIterations : int
        maximum number of iterations for the optimization (outer loop)
    past_non_improving : int
        number of steps without improvement before stopping
    progress_bar : bool
        whether to show a progress bar


    Returns
    -------
    Tuple[List, int, Optional[Tuple[np.ndarray, np.ndarray]]]
        [idx,h] return the vector of group membership of the nodes and the parameter h
    """
    # Compute necessary summaries from A
    n = A.shape[0]

    # use data driven h
    h = int(h) if h is not None else int(_oracle_analysis_badnwidth(A=A.astype(float)))

    # min h i s 2 so that no node is in its own group
    h = max(2, min(n, np.round(h)))

    lastGroupSize = n % h

    # step down h to avoid singleton group
    while lastGroupSize == 1 and h > 2:
        h = h - 1
        lastGroupSize = n % h

    idxInit = _first_guess_blocks(A, h)

    best_assignment = greedy_opt(
        A,
        idxInit,
        absTol=absTol,
        maxNumIterations=maxNumIterations,
        past_non_improving=past_non_improving,
        progress_bar=progress_bar,
    )
    return best_assignment, h


def _oracle_analysis_badnwidth(A: np.ndarray, type_: str = "degs", alpha: float = 1) -> float:
    c = min(4, np.sqrt(A.shape[0]) / 8)
    h, _ = oracbwplugin(A=A, c=c, type_=type_, alpha=alpha)
    return int(h)


def oracbwplugin(
    A: np.ndarray,
    c: float,
    type_: str = "degs",
    alpha: float = 1,
    num_diff_degrees: int = 100,
) -> Tuple[float, float]:
    """Oracle bandwidth plug-in estimtor for network histograms.

    The call h = oracbwplugin(A,c,type_,alpha) returns a plug-in estimate
    of the optimal histogram bandwidth (blockmodel community size) as a
    function of the following inputs

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (must be a simple graph)
    c : float
        positive multiplier by which to estimate slope 1/- sqrt(n)
    type_ : str
         Estimate slope from sorted vector ('degs' or 'eigs'). Defaults to "degs".
    alpha : float
         Holder exponent. Defaults to 1.
    num_diff_degrees : int
            Number of different degrees to use in slope estimation. Defaults to 100.
            If the number of nodes is smaller than num_diff_degrees, then all the nodes are used.

    Returns
    -------
    Tuple[float, float]
        optimal histogram bandwidth, estimated mean squared error

    Raises
    ------
    ValueError
        if c is not positive
    NotImplementedError
        alpha not 1
    NotImplementedError
        type_ not degs or eigs

    Examples
    --------
    >> h, _ = oracbwplugin(A,3,'eigs',1); # returns h = 73.5910

    >> h, _ = oracbwplugin(A,3,'degs',1); # returns h = 74.1031
    """
    # input checks
    if c <= 0:
        raise ValueError("c must be positive")
    if alpha != 1:
        raise NotImplementedError("Currently only support alpha = 1")

    # conversion for scipy functions
    A = A.astype(float)

    n = A.shape[0]
    num_diff_degrees = min(num_diff_degrees, n)
    midPt = np.arange(round(n / 2 - c * np.sqrt(n)) - 1, round(n / 2 + c * np.sqrt(n)))
    rhoHat = np.sum(A) / (n * (n - 1))
    pseudo_inverse_rho_hat = 1 / rhoHat if rhoHat != 0 else 0

    if type_ == "eigs":
        mult, u = scipy.sparse.linalg.eigs(A, 1, which="LR")
        u = u.real.ravel()
    elif type_ == "degs":
        u = np.sum(A, axis=1)
        mult = (u.T @ A @ u) / np.sum(u * u) ** 2
    else:
        raise NotImplementedError(f"Unknown input type_: {type_}")

    # if all the degrees are the same, there is no information in there
    if np.unique(u).size == 1:
        h = 1
        estMSqrd = 0
        logger.warning("All the degrees are the same, there is no information in there")
    # if the slope is 0, we have an issue: we need to have a non uniform array
    # of degrees to compute the slope: we augment the value of c to get a non uniform array
    # if needed
    else:
        u = np.sort(u.ravel())
        uMid = u[midPt]
        increment = 1
        while np.unique(uMid).size <= num_diff_degrees & np.min(midPt) > 0 & np.max(midPt) < n:
            midPt = np.arange(
                max(0, round(n / 2 - c * np.sqrt(n)) - 1 - increment),
                min(n - 1, round(n / 2 + c * np.sqrt(n)) + increment),
            )
            uMid = u[midPt]
            increment += 1
        reg = stats.linregress(np.arange(1, len(uMid) + 1), uMid)
        p = (reg[1], reg[0])
        h = (
            2 ** (alpha + 1)
            * alpha
            * mult**2
            * (p[0] + p[1] * len(uMid) / 2) ** 2
            * p[1] ** 2
            * pseudo_inverse_rho_hat
        ) ** (-1 / (2 * (alpha + 1)))
        estMSqrd = (
            2
            * mult**2
            * (p[0] + p[1] * len(uMid) / 2) ** 2
            * p[1] ** 2
            * pseudo_inverse_rho_hat**2
            * (n + 1) ** 2
        )
    # MISEfhatBnd = estMSqrd * ((2 / np.sqrt(estMSqrd)) * (sampleSize * rhoHat) ** (-1 / 2) + 1 / n)
    return h, estMSqrd


def _first_guess_blocks(A: np.ndarray, h: int) -> np.ndarray:
    """Compute the first guess of the node membership (block labels).

    Parameters
    ----------
    A : np.ndarray
        adjacency matrix
    h : int
        number of nodes per block

    Returns
    -------
    np.ndarray
        first guess of the node membership (block labels)
    """
    n = A.shape[0]

    if A.dtype != np.dtype("int"):
        logger.info("A is not int, casting to int for faster computation")
        distVec = fast_pairwise_distance(A.astype(int))
    else:
        distVec = fast_pairwise_distance(A)

    L = np.ones_like(distVec) - distVec**2
    d = np.sum(L, axis=1)
    d = d[:, np.newaxis]
    L_inter = np.outer(d**-0.5, d**-0.5) * L - np.outer(np.sqrt(d), np.sqrt(d)) / np.sqrt(
        np.sum(d**2)
    )
    _, u = scipy.sparse.linalg.eigs(L_inter, k=1, which="LR")
    u = u.real.ravel()

    # set 1st coord >= 0 wlog, to fix an arbitrary sign permutation
    u = u * np.sign(u[0])
    # sort on this embedding in ascending fashion
    ind = u.argsort().astype(int)
    k = int(np.ceil(n / h))

    # Assign initial label vector from row-similarity ordering
    # this is correct
    idxInit = np.zeros(n)
    for i in range(k):
        idxInit[ind[i * h : min(n, (i + 1) * h)]] = i
    return idxInit.astype(int)


@jit(nopython=True, parallel=False, fastmath=True)
def fast_pairwise_distance(A):
    """Compute the pairwise distance between each row of A.

    Parameters
    ----------
    A : np.ndarray
        adjacency matrix

    Returns
    -------
    np.ndarray
        pairwise distance between each row of A
    """
    n = A.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = np.sum(np.abs(A[i] - A[j])) / n
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric matrix
    return dist_matrix
