from typing import List, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.special
from scipy import stats
from sklearn.metrics import pairwise_distances

from pygraphon.utils.utils_graph import check_simple_adjacency_matrix

from .graphest_fastgreedy import graphest_fastgreedy


def oracle_analysis_badnwidth(A: np.ndarray, type: str = "degs", alpha: float = 1) -> float:
    c = min(4, np.sqrt(A.shape[0]) / 8)
    h, _ = oracbwplugin(A=A, c=c, type=type, alpha=alpha)
    return int(h)


def oracbwplugin(
    A: np.ndarray, c: float, type: str = "degs", alpha: float = 1
) -> Tuple[float, float]:
    """Oracle bandwidth plug-in estimtor for network histograms
    h = oracbwplugin(A,c,type,alpha) returns a plug-in estimate
    of the optimal histogram bandwidth (blockmodel community size) as a
    function of the following inputs

     Args:
         A (np.ndarray): Adjacency matrix (must be a simple graph)
         c (float): positive multiplier by which to estimate slope 1/- sqrt(n)
         type (str, optional): Estimate slope from sorted vector ('degs' or 'eigs'). Defaults to "degs".
         alpha (float, optional): Holder exponent. Defaults to 1.

     Returns:
         float: h


     Examples:
         >> h = oracbwplugin(A,3,'eigs',1); # returns h = 73.5910
         >> h = oracbwplugin(A,3,'degs',1); # returns h = 74.1031
    """

    # input checks
    if c <= 0:
        raise ValueError("c must be positive")
    if alpha != 1:
        raise NotImplementedError("Currently only support alpha = 1")
    check_simple_adjacency_matrix(A)

    # conversion for scipy functions
    A = A.astype(float)

    n = A.shape[0]
    midPt = np.arange(round(n / 2 - c * np.sqrt(n) - 1), round(n / 2 + c * np.sqrt(n)))
    rhoHat = np.sum(A) / (n * (n - 1))
    pseudo_inverse_rho_hat = 1 / rhoHat if rhoHat != 0 else 0

    if type == "eigs":
        mult, u = scipy.sparse.linalg.eigs(A, 1, which="LR")
        u = u.real.ravel()
    elif type == "degs":
        u = np.sum(A, axis=1)
        mult = (u.T @ A @ u) / np.sum(u * u) ** 2
    else:
        raise NotImplementedError(f"Unknown input type: {type}")

    u = np.sort(u.ravel())
    uMid = u[midPt]
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


def first_guess_blocks(A: np.ndarray, h: int, regParam: float) -> np.ndarray:
    """
    This function is used to compute the first guess of the block labels.
    """
    n = A.shape[0]

    if regParam == 0:
        distVec = pairwise_distances(A, A, metric="manhattan") / n
    else:
        A_inter = A + regParam * np.ones_like(A) * regParam
        distVec = pairwise_distances(A_inter, A_inter, metric="manhattan") / n
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
    return idxInit


def nethist(
    A: np.ndarray, h: int = None, verbose: bool = False, trace: bool = False
) -> Tuple[List, int, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Computes the network histogram of an N-by-N
    adjacency matrix, which must be 0-1 valued, symmetric, and with zero
    main diagonal.

    Args:
        A (np.ndarray): adjacency matrix
        h (int, optional): specifies the number of nodes in each histogram bin,
        which is automatically determined if h is None. Defaults to None.

    Returns:
        Tuple[List,int]: [idx,h] return the vector of group membership of the nodes and the parameter h
    """
    check_simple_adjacency_matrix(A)

    # conversion for scipy functions
    A = A.astype(float)

    # Compute necessary summaries from A
    n = A.shape[0]
    rhoHat = np.sum(A) / (n * (n - 1))

    # use data driven h
    h = int(h) if h is not None else oracle_analysis_badnwidth(A)

    # min h i s 2 so that no node is in its own group
    h = max(2, min(n, np.round(h)))

    lastGroupSize = n % h

    # step down h to avoid singleton group
    while lastGroupSize == 1 and h > 2:
        h = h - 1
        lastGroupSize(n % h)

    idxInit = first_guess_blocks(A, h, regParam=rhoHat / 4)
    if trace:
        idx, k, trace = graphest_fastgreedy(
            A=A, hbar=h, inputLabelVec=idxInit, verbose=verbose, trace=trace
        )
    else:
        idx, k = graphest_fastgreedy(
            A=A, hbar=h, inputLabelVec=idxInit, verbose=verbose, trace=trace
        )
    return idx, h, trace
