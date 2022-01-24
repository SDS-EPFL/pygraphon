from typing import List, Tuple

import numpy as np
import numpy.random as rnd
import scipy
import scipy.sparse.linalg
import scipy.special
from scipy.spatial.distance import pdist, squareform

MATLAB_EPS = np.spacing(1)


def upper_triangle_values(array):
    return array[np.triu_indices(array.shape[0])]


def first_guess_blocks(A: np.ndarray, h: int, regParam: float) -> np.ndarray:
    """
    This function is used to compute the first guess of the block labels.
    """
    n = A.shape[0]
    distVec = pdist(A + regParam, "hamming")
    L = 1 - squareform(distVec)
    d = np.sum(L, axis=1)
    d = d[:, np.newaxis]
    L_inter = (d ** -0.5 @ np.transpose(d ** -0.5)) * L - np.sqrt(d) @ np.transpose(
        np.sqrt(d)
    ) / np.sqrt(d.T @ d)
    _, u = scipy.sparse.linalg.eigs(L_inter, k=1, which="SM")
    u = np.ravel(u.real)

    # set 1st coord >= 0 wlog, to fix an arbitrary sign permutation
    u = u * np.sign(u[0])
    # sort on this embedding in ascending fashion
    ind = u.argsort()[::-1]

    k = int(np.ceil(n / h))

    # Assign initial label vector from row-similarity ordering
    idxInit = np.zeros(n)
    for i in range(k):
        idxInit[ind[i * h : min(n, (i + 1) * h)]] = i

    # idxInit = np.array([1,1,1,0,0,0])
    return idxInit


def first_guess_blocks_python(A: np.ndarray, h: int, *args, **kwargs) -> np.ndarray:

    n = A.shape[0]
    laplacian = scipy.sparse.csgraph.laplacian(A, normed=True)
    _, u = scipy.sparse.linalg.eigs(laplacian, k=1, which="SM")
    u = np.ravel(u.real)

    # set 1st coord >= 0 wlog, to fix an arbitrary sign permutation
    u = u * np.sign(u[0])
    # sort on this embedding in ascending fashion
    ind = u.argsort()[::-1]

    k = int(np.ceil(n / h))

    # Assign initial label vector from row-similarity ordering
    idxInit = np.zeros(n)
    h = int(h)
    for i in range(k):
        idxInit[ind[i * h : min(n, (i + 1) * h)]] = i

    return idxInit.astype(int)


def nethist(A: np.ndarray, h: int = None) -> Tuple[List, int]:
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

    #idxInit = first_guess_blocks(A, h, regParam=rhoHat / 4)
    idxInit = first_guess_blocks_python(A, h, rhoHat / 4)

    print(idxInit)
    idx, k = graphest_fastgreedy(A=A, hbar=h, inputLabelVec=idxInit)
    return idx, h


def oracbwplugin(
    A: np.ndarray, c: float, type: str = "degs", alpha: float = 1
) -> Tuple[float, float]:
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
    # sampleSize = scipy.special.comb(n, 2)
    rhoHat = np.sum(A) / (n * (n - 1))

    if type == "eigs":
        mult, u = scipy.sparse.linalg.eigs(A, 1)
        u = u.real
    elif type == "degs":
        u = np.sum(A, axis=1)
        mult = (u.T @ A @ u) / (u.T @ u) ** 2
    else:
        raise NotImplementedError(f"Unknown input type: {type}")

    pseudo_inverse_rho_hat = 1 / rhoHat if rhoHat != 0 else 1
    u = np.sort(u.ravel())
    uMid = u[midPt[0] : midPt[-1]]
    p = np.polyfit(range(1, len(u) + 1), u, 1)
    h = (
        2 ** (alpha + 1)
        * alpha
        * mult ** 2
        * (p[1] + p[0] * len(uMid) / 2) ** 2
        * p[1] ** 2
        * pseudo_inverse_rho_hat
    ) ** (-1 / (2 * (alpha + 1)))
    estMSqrd = (
        2
        * mult ** 2
        * (p[1] + p[1] * len(uMid) / 2) ** 2
        * p[0] ** 2
        * pseudo_inverse_rho_hat ** 2
        * (n + 1) ** 2
    )
    # MISEfhatBnd = estMSqrd * ((2 / np.sqrt(estMSqrd)) * (sampleSize * rhoHat) ** (-1 / 2) + 1 / n)
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
        numGreedySteps = scipy.special.comb(A.shape[0], 2, exact=True)
    else:
        numGreedySteps = 2 * 10 ** 4

    n = A.shape[0]
    sampleSize = scipy.special.comb(n, 2)
    smaller_last_group = 1 if n // hbar == 0 else 0
    k = int(np.ceil(n / hbar))
    equalSizeInds = np.arange(1, (k - smaller_last_group))
    orderedLabels = np.zeros((n, 1))
    h = np.zeros(k)
    hbar = int(hbar)
    orderedClusterInds = np.zeros((k, hbar))
    for a in range(1, k - smaller_last_group + 1):
        orderedInds = np.arange((a - 1) * hbar + 1, a * hbar + 1)
        h[a - 1] = len(orderedInds)
        orderedLabels[orderedInds - 1] = a
        orderedClusterInds[a - 1, :] = orderedInds

    if smaller_last_group:
        orderedIndsLast = np.arange((k - 1) * hbar + 1, n)
        h[k - 1] = len(orderedInds)
        orderedClusterInds[k - 1, :] = np.concatenate(
            orderedIndsLast, np.zeros(hbar - len(orderedIndsLast))
        )

    habSqrd = np.outer(h, h) - np.diag(np.multiply(h, h) - np.multiply(h, (h - 1)) / 2)

    if not np.all(habSqrd) >= 1:
        raise RuntimeError("All clusters must contain at least 2 nodes")
    if np.max(orderedClusterInds[k - 1, :]) != n:
        raise RuntimeError("All nodes must be assigned to a cluster")
    if np.sum(h) != n:
        raise RuntimeError("All nodes must be assigned to a cluster")

    initialLabelVec = inputLabelVec.astype(int)
    initialClusterInds = np.zeros((k, hbar), dtype=int)
    initialClusterCentroids = np.zeros((k, n))

    for a in range(k - smaller_last_group):
        initialClusterInds[a, :] = np.where(initialLabelVec == a)[0]
        initialClusterCentroids[a, :] = np.sum(A[:, initialClusterInds[a, :]], axis=1)

    if smaller_last_group:
        initialClusterInds[k, 1 : len(np.where(initialLabelVec == k)[0])] = np.where(
            initialLabelVec == k
        )[0]

    initialACounts = getSampleCounts(A, initialClusterInds)
    initialLL = fastNormalizedBMLogLik(
        upper_triangle_values(initialACounts) / upper_triangle_values(habSqrd),
        upper_triangle_values(habSqrd),
        sampleSize,
    )

    bestLL = initialLL
    oldNormalizedBestLL = bestLL * 2 * sampleSize / np.sum(A)
    bestLabelVec = initialLabelVec
    bestCount = 0
    conseczeroImprovement = 0
    tolCounter = 0

    for mm in range(maxNumRestarts):
        oneTwoVec = np.array(rnd.uniform(size=numGreedySteps) > 2 / 3) + np.ones(
            numGreedySteps, dtype=int
        )

        # random integers between 0 and n-1
        iVec = np.ceil(rnd.uniform(size=numGreedySteps) * n - 1).astype(int)
        kVec = np.ceil(rnd.uniform(size=numGreedySteps) * n - 1).astype(int)
        jVec = np.ceil(rnd.uniform(size=numGreedySteps) * n - 1).astype(int)

        bestClusterInds = np.zeros(shape=(k, hbar), dtype=int)
        for a in range(k - smaller_last_group):
            bestClusterInds[a, :] = np.where(bestLabelVec == a)[0]

        if smaller_last_group:
            bestClusterInds[k, 1 : len(np.where(bestLabelVec == k)[0])] = np.where(
                np.where(bestLabelVec == k)[0]
            )

        bestACounts = getSampleCounts(A, bestClusterInds)
        bestLL = fastNormalizedBMLogLik(
            upper_triangle_values(bestACounts) / upper_triangle_values(habSqrd),
            upper_triangle_values(habSqrd).ravel(),
            sampleSize,
        )

        currentACounts = bestACounts
        currentClusterInds = bestClusterInds
        currentLL = bestLL
        currentLabelVec = bestLabelVec

        for m in range(numGreedySteps):

            # prepare to update quantities for trial clustering
            trialClusterInds = currentClusterInds
            trialLabelVec = currentLabelVec
            trialACounts = currentACounts
            trialLL = currentLL

            # implement consecutive pairwise swaps to obtain trial clustering
            for swapNum in range(oneTwoVec[m]):
                if swapNum == 0:
                    # ideally here i,j are very similar, but in different groups
                    i = iVec[m]
                    j = jVec[m]
                    # get group labels of nodes in chosen pair
                    a = trialLabelVec[i]
                    b = trialLabelVec[j]

                # check that the pairwise swap was made
                elif a != b:
                    i = jVec[m]
                    j = kVec[m]
                    a = trialLabelVec[i]
                    b = trialLabelVec[j]

                # swap and update the trial likelihood only if nodes i and j are in different clusters
                if a != b:

                    trialLabelVec[i], trialLabelVec[j] = b, a

                    habSqrdCola = habSqrd[:, a]
                    habSqrdColb = habSqrd[:, b]
                    habSqrdEntryab = habSqrd[a, b]

                    oldThetaCola = trialACounts[:, a] / habSqrdCola
                    oldThetaColb = trialACounts[:, b] / habSqrdColb
                    oldThetaEntryab = np.array(trialACounts[a, b] / habSqrdEntryab)

                    oldThetaCola, oldThetaColb, oldThetaEntryab = bound_away_from_one_and_zeros(
                        [oldThetaCola, oldThetaColb, oldThetaEntryab]
                    )

                    # begin updating
                    trialClusterInds[a, trialClusterInds[a, :] == i] = j
                    trialClusterInds[b, trialClusterInds[b, :] == j] = i
                    ARowiMinusRowj = A[i, :] - A[j, :]

                    # concatenate all possible group indices into a a matrix
                    clusterIndMat = trialClusterInds[equalSizeInds, :]
                    sumAijc = np.sum(ARowiMinusRowj[clusterIndMat], axis=1)
                    trialACounts[equalSizeInds, a] = trialACounts[equalSizeInds, a] - sumAijc
                    trialACounts[equalSizeInds, b] = trialACounts[equalSizeInds, b] + sumAijc

                    if smaller_last_group:  # take care of last group separately if unequal size
                        sumAijEnd = np.sum(
                            ARowiMinusRowj[trialClusterInds[k, trialClusterInds[k, :] > 0]]
                        )
                        trialACounts[k, a] = trialACounts[k, a] - sumAijEnd
                        trialACounts[k, b] = trialACounts[k, b] + sumAijEnd

                    # update the above for special case c==a | c==b
                    trialACounts[a, a] = trialACounts[a, a] + A[i, j]
                    trialACounts[b, b] = trialACounts[b, b] + A[i, j]
                    if smaller_last_group and b == k:
                        trialACounts[a, b] = (
                            trialACounts[a, b]
                            - np.sum(
                                ARowiMinusRowj[trialClusterInds[b, trialClusterInds[b, :] > 0]]
                            )
                            - 2 * A[i, j]
                        )
                    else:
                        trialACounts[a, b] = (
                            trialACounts[a, b]
                            - np.sum(ARowiMinusRowj[trialClusterInds[b, :]])
                            - A[i, j]
                        )
                    trialACounts[b, a] = trialACounts[a, b]

                    # Normalize and respect symmetry of trialAbar matrix
                    trialACounts[a, :] = trialACounts[:, a]
                    trialACounts[b, :] = trialACounts[:, b]

                    # Now calculate changed likelihood directly
                    thetaCola = trialACounts[:, a] / habSqrdCola
                    thetaColb = trialACounts[:, b] / habSqrdColb
                    thetaEntryab = np.array(trialACounts[a, b] / habSqrdEntryab)

                    thetaCola, thetaColb, thetaEntryab = bound_away_from_one_and_zeros(
                        [thetaCola, thetaColb, thetaEntryab]
                    )

                    # For this to work, we will have had to subtract out terms prior to updating
                    deltaNegEnt = delta_neg(
                        habSqrdCola, thetaCola, habSqrdColb, thetaColb, habSqrdEntryab, thetaEntryab
                    )
                    oldDeltaNegEnt = delta_neg(
                        habSqrdCola,
                        oldThetaCola,
                        habSqrdColb,
                        oldThetaColb,
                        habSqrdEntryab,
                        oldThetaEntryab,
                    )

                    # Update the log-likelihood - O(k)
                    trialLL = trialLL + (deltaNegEnt - oldDeltaNegEnt) / sampleSize

            # Metropolis or greedy step; if trial clustering accepted, then update current <- trial

            if trialLL > currentLL:
                currentLabelVec = trialLabelVec
                currentLL = trialLL
                currentACounts = trialACounts
                currentClusterInds = trialClusterInds

        if currentLL > bestLL:
            bestLL = currentLL
            bestLabelVec = currentLabelVec
            bestCount += 1

        if mm // 5 == 0:
            normalizedBestLL = bestLL * 2 * sampleSize / np.sum(A)
            if bestCount == 0:
                conseczeroImprovement += 1
            if normalizedBestLL - oldNormalizedBestLL < absTol:
                tolCounter += 1
            else:
                tolCounter = 0
            oldNormalizedBestLL = normalizedBestLL
            # if 3 consecutive likelihood improvements less than specified tolerance break
            if tolCounter > 3:
                print("tolerance counter break")
                break
            if allInds == 1:
                # local optimum likely reached in random-ordered greedy like likelihood search
                if conseczeroImprovement == 2:
                    print("consecutive zero improvement break")
                    break
            else:
                # Local optimum likely reached in random ordere greedy like likelihood search
                if conseczeroImprovement == np.ceil(k * scipy.special.binom(n, 2) / numGreedySteps):
                    print("consecutive zero improvement break, version 2")
                    break

    return bestLabelVec, k


def delta_neg(habSqrdCola, thetaCola, habSqrdColb, thetaColb, habSqrdEntryab, thetaEntryab):

    habSqrdCola = np.array(habSqrdCola)
    thetaCola = np.array(thetaCola)
    habSqrdColb = np.array(habSqrdColb)
    thetaColb = np.array(thetaColb)
    habSqrdEntryab = np.array(habSqrdEntryab)
    thetaEntryab = np.array(thetaEntryab)

    return np.sum(
        np.multiply(
            habSqrdCola,
            np.multiply(
                thetaCola,
                np.log(thetaCola)
                + np.multiply(
                    np.ones_like(thetaCola) - thetaCola, np.log(np.ones_like(thetaCola) - thetaCola)
                ),
            ),
        )
        - np.multiply(
            habSqrdColb,
            np.multiply(
                thetaColb,
                np.log(thetaColb)
                + np.multiply(
                    np.ones_like(thetaColb) - thetaColb, np.log(np.ones_like(thetaColb) - thetaColb)
                ),
            ),
        )
        - np.multiply(
            habSqrdEntryab,
            np.multiply(
                thetaEntryab,
                np.log(thetaEntryab)
                + np.multiply(
                    np.ones_like(thetaEntryab) - thetaEntryab,
                    np.log(np.ones_like(thetaEntryab) - thetaEntryab),
                ),
            ),
        )
    )


def bound_away_from_one_and_zeros(arrays: List[np.ndarray]):
    """
    This function is used to bound away from 1 and 0 in the log likelihood.
    This is done to avoid numerical issues.
    """
    for array in arrays:
        array[array <= 0] = MATLAB_EPS
        array[array >= 1] = 1 - MATLAB_EPS
    return arrays


def getSampleCounts(X, clusterInds):

    numClusters = clusterInds.shape[0]
    Xsums = np.zeros((numClusters, numClusters), dtype=int)
    for b in range(1, numClusters):
        for a in range(b):  # sum over strict upper-triangular elements
            clusterIndsa = clusterInds[a, clusterInds[a, :] >= 0]
            clusterIndsb = clusterInds[b, clusterInds[b, :] >= 0]
            Xsums[a, b] = np.sum(X[clusterIndsa, :][:, clusterIndsb])
    Xsums = Xsums + Xsums.T
    for a in range(numClusters):  # relies on A begin symmetric and with no self-loops
        clusterIndsa = clusterInds[a, clusterInds[a, :] >= 0]
        Xsums[a, a] = np.sum(X[clusterIndsa, :][:, clusterIndsa]) / 2

    return Xsums


def fastNormalizedBMLogLik(thetaVec: np.ndarray, habSqrdVec: np.ndarray, sampleSize: int) -> float:
    thetaVec = thetaVec.astype(float)
    thetaVec[thetaVec <= 0] = MATLAB_EPS
    thetaVec[thetaVec >= 1] = 1 - MATLAB_EPS
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
