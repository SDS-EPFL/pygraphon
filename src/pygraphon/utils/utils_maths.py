from typing import Iterable
from itertools import permutations


def generate_all_permutations(size: int = 3) -> Iterable:
    """Generate all permutations of a given size.

    Args:
        size (int, optional): size of the permutation (0,1,2,...,size-1). Defaults to 3.

    Returns:
        Iterable: all permutations of the given size
    """

    return permutations(range(size))

