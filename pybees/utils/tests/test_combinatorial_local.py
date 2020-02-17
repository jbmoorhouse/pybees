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

np.random.seed(0)

def test_swap():
    # simple test
    bee_permutation = np.arange(10)
    x = _random_choice_unique(np.arange(160).reshape(16, -1))

    assert swap(bee_permutation, x.shape[0]) == x


