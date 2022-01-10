from pygraphon.graphons.StepGraphon import StepGraphon
import numpy as np

from pygraphon.utils.utils_maths import generate_all_permutations
from pygraphon.utils.utils_matrix import permute_matrix


def distance_StepGraphon(
    graphon1: StepGraphon, graphon2: StepGraphon, norm: str = "MISE", exchangeable: bool = True
) -> float:
    """Implement the mean squared error and mean absolute error for stepgraphons of same size (same number of blocks and same areas of blocks)


    Args:
        graphon1 (StepGraphon): first stepgraphon to compare
        graphon2 (StepGraphon): second stepgraphon to compare
        norm (str, optional): in ["MAE","MISE"]. Defaults to "MISE".
        exchangeable (bool, optional): if sets to true, the norm will try all possible permutations of the blocks to find the lowest distance. 
        Otherwise assume correspondance between the blocks of the first and second graphon. Defaults to True.

    Raises:
        NotImplementedError: if the two graphons are not of the same size (different number of blocks or heteogeneous size of blocks)
        ValueError: if norm not in MAE or MISE

    Returns:
        float: distance between the two graphons
    """

    # get the data we need
    graphon1_matrix = graphon1.graphon
    graphon2_matrix = graphon2.graphon

    # check that graphons have the same bandwidth for their blocks
    #  this means they have the same size of blocks if the graphon have the same shape
    if graphon2.bandwidthHist != graphon1.bandwidthHist:
        raise NotImplementedError("different size of graphons cannot be compared for now")

    # check if the graphon blocks all have same size
    if len(np.unique(graphon1.areas)) != 1 or len(np.unique(graphon2.areas)) != 1:
        raise NotImplementedError("Cannot compare graphons with heterogeneous block sizes")

    # generate all possible permutations
    permutations_possible = generate_all_permutations(graphon1_matrix.shape[0])

    norm_value = np.sqrt(
                np.sum(((graphon1 - graphon2_matrix) ** 2) * graphon1.areas)
            )
    if not exchangeable:
        return norm_value
    for sigma in permutations_possible:
        if norm == "MISE":
            result = np.sqrt(
                np.sum(((graphon1 - permute_matrix(graphon2_matrix, sigma)) ** 2) * graphon1.areas)
            )
        elif norm in ["ABS", "MAE"]:
            result = np.average(
                np.sum(np.abs(graphon1 - permute_matrix(graphon2_matrix, sigma)) * graphon1.areas)
            )
        else:
            raise ValueError(f"norm not defined, got {norm}")
        norm_value = min(norm_value, result)

    return norm_value
