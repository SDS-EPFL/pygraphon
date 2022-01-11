from typing import List, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg
import scipy.special
from numpy import linalg
from numpy.linalg import pinv
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform


def nethist(A: np.ndarray, h: int = None) -> Tuple[List, int]:
    """Computes the network histogram of an N-by-N
    adjacency matrix, which must be 0-1 valued, symmetric, and with zero
    main diagonal.

    Args:
        A (np.ndarray): adjacency matrix
        h (int, optional): specifies the number of nodes in each histogram bin, which is automatically determined if h is None. Defaults to None.

    Returns:
        Tuple[List,int]: [idx,h] return the vector of group membership of the nodes and the parameter h
    """

    A = checkAdjacencyMatrix(A)

    # Compute necessary summaries from A
    n = A.shape[0]
    rhoHat = np.sum(A) / (n * (n - 1))

    # use data driven h
    if h is None:
        c = min(4, np.sqrt(n) / 8)
        # fractional h before rounding
        h, _ = oracbwplugin(A=A, c=c, type="degs", alpha=1)

    # why do we have minimum h = 2 ?
    h = max(2, min(n, np.round(h)))

    lastGroupSize = n % h

    # step down h to avoid singleton group
    while lastGroupSize == 1 and h > 2:
        h = h - 1
        lastGroupSize(n % h)

    regParam = rhoHat / 4

    distVec = pdist(A + regParam, "hamming")
    L = 1 - squareform(distVec)
    d = np.sum(L, axis=1)
    d = d[:, np.newaxis]
    L_inter = (d ** -0.5 @ np.transpose(d ** -0.5)) * L - np.sqrt(d) @ np.transpose(
        np.sqrt(d)
    ) / np.sqrt(d.T @ d)
    _, u = scipy.sparse.linalg.eigs(L_inter, k=1)
    u = u.real

    # set 1st coord >= 0 wlog, to fix an arbitrary sign permutation
    u = u * np.sign(u[0])
    # sort on this embedding in ascending fashion
    ind = u.argsort()[::-1]

    k = np.ceil(n / h)

    idxInit = np.zeros((n, 1))
    for i in range(1, k):
        idxInit[ind[(i - 1) * h + 1 : min(n, i * h)]] = i

    idx, k = graphest_fastgreedy(A=A, hbar=h, inputLabelVec=idxInit)
    return idx, h
    

def oracbwplugin(
    A: np.ndarray, c: float, type: str = "degs", alpha: float = 1
) -> float:
    """Oracle bandwidth plug-in estimtor for network histograms
    h = oracbwplugin(A,c,type,alpha) returns a plug-in estimate
    of the optimal histogram bandwidth (blockmodel community size) as a
    function of the following inputs

     Args:
         A (np.ndarray): Adjacency matrix (must be a simple graph)
         c (float): positive multiplier by which to estumate slope 1/- sqrt(n)
         type (str, optional): Estimate slope from sorted vector ('degs' or 'eigs'). Defaults to "degs".
         alpha (float, optional): Holder exponent. Defaults to 1.

     Returns:
         float: h



     Examples:
         >> h = oracbwplugin(A,3,'eigs',1); # returns h = 73.5910

         >> h = oracbwplugin(A,3,'degs',1); # returns h = 74.1031
    """

    # input checks
    assert c > 0, "c must be positive"
    A = checkAdjacencyMatrix(A)

    n = A.shape[0]
    midPt = np.arange(round(n / 2 - c * np.sqrt(n)), round(n / 2 + c * np.sqrt(n)))
    sampleSize = scipy.special.comb(n, 2)
    rhoHat = np.sum(A) / (n * (n - 1))

    if type == "eigs":
        mult, u = scipy.sparse.linalg.eigs(A, 1)
        u = u.real
    elif type == "degs":
        u = np.sum(A, axis=1)
        mult = (u.T @ A @ u) / (u.T @ u) ** 2
    else:
        raise NotImplementedError(f"Unknown input type: {type}")

    u = np.sort(u.ravel())
    uMid = u[midPt[0] : midPt[-1]]
    p = np.polyfit(range(1, len(u) + 1), u, 1)
    h = (
        2
        ^ (alpha + 1) * alpha * mult
        ^ 2 * (p[1] + p[0] * len(uMid) / 2)
        ^ 2 * p(1)
        ^ 2 * pinv(rhoHat)
    ) ^ (-1 / (2 * (alpha + 1)))
    estMSqrd = (
        2 * mult
        ^ 2 * (p[1] + p(1) * len(uMid) / 2)
        ^ 2 * p[0]
        ^ 2 * pinv(rhoHat)
        ^ 2 * (n + 1)
        ^ 2
    )
    MISEfhatBnd = estMSqrd * (
        (2 / np.sqrt(estMSqrd)) * (sampleSize * rhoHat) ^ (-1 / 2) + 1 / n
    )
    return h, estMSqrd


def graphest_fastgreedy(
    A: np.ndarray, hbar: int, inputLabelVec: List, absTol=2.5 * 1e-4, maxNumRestarts=500
):
    """Implements likelihood-based optimization for nethist.

    Args:
        A (np.ndarray): Adjacency matrix (simple graph)
        hbar (int): [description]
        inputLabelVec (List): [description]

    Returns:
        [type]: [description]
    """

    # arbitrary size cutoff
    #  Consider basing this choice on size (or sparse-vs-dense storage) of A
    if A.shape[0] <= 256:
        allInds = 1
    else:
        # Only a random subset of pairs will be visited on each iteration
        allInds = 0
    if allInds:
        numGreedySteps = scipy.special.comb(A.shape[0],2)
    else:
        numGreedySteps = 2*10**4

    n = A.shape[0]
    sampleSize = scipy.special.comb(n,2)

    raise NotImplementedError
    idx = None
    k = None
    return idx, k


def getSampleCounts(X, clusterInds):
    raise NotImplementedError


def fastNormalizedBMLogLik(
    thetaVec: np.ndarray, habSqrdVec: float, sampleSize: int
) -> float:
    thetaVec[thetaVec <= 0] = np.spacing(1)
    thetaVec[thetaVec >= 1] = np.spacing(1)
    negEntVec = thetaVec * np.log(thetaVec) + (1 - thetaVec) * np.log(1 - thetaVec)
    normLogLik = sum(habSqrdVec * negEntVec) / sampleSize
    return normLogLik


def checkAdjacencyMatrix(A: np.ndarray, rtol=1e-05, atol=1e-08):
    if np.sum(np.abs(np.diag(A))) > 0:
        raise ValueError("No self loops allowed")

    if not np.array_equal(A, A.astype(bool)):
        raise ValueError("Only simple graphs allowed, entries should be binary")

    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix should be square")

    if A.shape[0] < 2:
        raise ValueError("Matrix input A must be of dimension at least 2 x 2")

    if not np.allclose(A, A.T, rtol=rtol, atol=atol):
        raise ValueError("Matrix input A should be symmetric")

    # change type of matrix to allow scipy functions to work
    return A.astype(float)
