import pytest
import numpy as np

from pybees.utils.combinatorial_search import (
    _random_choice_unique,
    swap,
    reversion,
    insertion
)

# =============================================================================
# Local search tests
# =============================================================================

def test_swap():

    # simple test with np.random.seed(0)
    # -------------------------------------------------------------------------
    np.random.seed(0)
    bee_permutation = np.arange(10)
    res = swap(bee_permutation, 16)

    # swapped arguments and known solution with np.random.seed(0)
    args = np.argwhere(res != bee_permutation)[:, 1].reshape(-1, 2)
    sol = np.array([[4, 9], [4, 6], [4, 6], [0, 4], [3, 7], [3, 5], [7, 9],
                    [5, 9], [2, 7], [7, 9], [4, 8], [3, 5], [6, 9], [2, 3],
                    [1, 6],[4, 9]])

    assert np.array_equal(args, sol)

