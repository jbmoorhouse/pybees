import pytest
import numpy as np

from pybees.utils.combinatorial_search import (
    _random_choice_unique,
    swap,
    reversion,
    insertion,
    _prepare_array
)

# =============================================================================
# Results and indices
# =============================================================================

bee_permutation, n_bees = np.arange(5),  10

funcs, test_results = [swap, reversion, insertion], {}

for func in funcs:
    func_name = func.__name__

    np.random.seed(0)
    res_one = func(bee_permutation, n_bees)
    res_two = func(bee_permutation, n_bees)

    test_results[f"{func_name}_1"] = (bee_permutation, res_one)
    test_results[f"{func_name}_2"] = (bee_permutation, res_two)

def helper(func, arr, solution):
    # simple test with np.random.seed(0)
    # -------------------------------------------------------------------------
    res = func(arr, 10)

    assert np.array_equal(res, solution)

# =============================================================================
# Local search tests
# =============================================================================

# swap
# -----------------------------------------------------------------------------
np.random.seed(0)

@pytest.mark.parametrize(
    "arr, solution", 
    [test_results['swap_1'], test_results['swap_2']])
def test_swap(arr, solution):
    helper(swap, arr, solution)

# reversion
# -----------------------------------------------------------------------------
np.random.seed(0)

@pytest.mark.parametrize(
    "arr, solution", 
    [test_results['reversion_1'], test_results['reversion_2']])
def test_reversion(arr, solution):
    helper(reversion, arr, solution)

# insertion
# -----------------------------------------------------------------------------
np.random.seed(0)

@pytest.mark.parametrize(
    "arr, solution", 
    [test_results['insertion_1'], test_results['insertion_2']])
def test_insertion(arr, solution):
    helper(insertion, arr, solution)
