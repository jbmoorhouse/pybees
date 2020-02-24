from pybees.utils.combinatorial_search import GlobalSearch
from scipy.spatial.distance import cdist
import numpy as np
import pytest

np.random.seed(0)

# =============================================================================
# Test correct inputs
# =============================================================================

@pytest.mark.parametrize(
    "correct_input", 
    [np.random.randint(10, size=[10, 3]),
     np.random.randint(10, size=[10, 2])]
)
def test_correct_input(correct_input):
    gs = GlobalSearch(correct_input)
    
    assert np.allclose(gs.coordinates, correct_input)
    assert np.allclose(gs.distance, cdist(correct_input, correct_input))

# =============================================================================
# Test bad inputs
# =============================================================================

@pytest.mark.parametrize(
    "bad_shape", 
    [np.random.randint(10, size=[10, 2, 1]),
     np.random.randint(10, size=[10])]
)
def test_bad_input_shape(bad_shape):
    msg = "XA must be a 2-dimensional array."

    with pytest.raises(ValueError, match = msg):
        GlobalSearch(bad_shape)

@pytest.mark.parametrize(
    "bad_element", 
    [np.array([["",2]]),
     np.array([[2,""]])]
)
def test_bad_elements(bad_element):
    msg = "could not convert string to float:"

    with pytest.raises(ValueError, match = msg):
        GlobalSearch(bad_element)


@pytest.mark.parametrize("bad_type", ["", 1, 1., (), []])
def test_bad_input_type(bad_type):
    msg = "XA must be a 2-dimensional array"

    with pytest.raises(ValueError, match = msg):
        GlobalSearch(bad_type)

# =============================================================================
# Test permute
# =============================================================================

np.random.seed(0)
GS = GlobalSearch(np.random.randint(10, size=[10, 2]))

@pytest.mark.parametrize("permute, correct_output", 
[(GS.permute(5), np.array([[3, 1, 2, 9, 5, 8, 4, 6, 0, 7],
        [1, 3, 7, 6, 9, 5, 2, 8, 0, 4],
        [1, 7, 8, 0, 3, 4, 2, 6, 9, 5],
        [0, 4, 3, 5, 6, 8, 7, 1, 2, 9],
        [0, 8, 7, 2, 1, 6, 4, 5, 3, 9]])),
 (GS.permute(5), np.array([[6, 4, 0, 1, 2, 8, 3, 5, 7, 9],
        [2, 9, 6, 4, 3, 5, 7, 8, 0, 1],
        [4, 9, 2, 7, 3, 1, 5, 8, 0, 6],
        [6, 4, 2, 8, 1, 3, 7, 0, 9, 5],
        [5, 1, 7, 4, 2, 3, 9, 8, 6, 0]]))]
 )
def test_permute(permute, correct_output):
    assert np.allclose(permute, correct_output)


@pytest.mark.parametrize("permute_bad_input", ["", [], 1.0])
def test_permute_bad_inputs(permute_bad_input):
    msg = ".* object cannot be interpreted as an integer"

    with pytest.raises(TypeError, match = msg): 
        GS.permute(permute_bad_input)

# =============================================================================
# Test nearest_neighbor
# =============================================================================

@pytest.mark.parametrize("nearest_neighbor, correct_output", 
[(GS.nearest_neighbor(5), np.array([[6, 2, 8, 5, 3, 4, 1, 0, 9, 7],
        [9, 0, 1, 4, 3, 7, 5, 8, 6, 2],
        [8, 5, 6, 2, 3, 4, 1, 0, 9, 7],
        [0, 9, 5, 8, 6, 2, 3, 4, 1, 7],
        [8, 5, 6, 2, 3, 4, 1, 0, 9, 7]])),
 (GS.nearest_neighbor(5), np.array([[5, 8, 6, 2, 3, 4, 1, 0, 9, 7],
        [9, 0, 1, 4, 3, 7, 5, 8, 6, 2],
        [0, 9, 5, 8, 6, 2, 3, 4, 1, 7],
        [9, 0, 1, 4, 3, 7, 5, 8, 6, 2],
        [6, 2, 8, 5, 3, 4, 1, 0, 9, 7]]))]
)
def test_nearest_neighbor(nearest_neighbor, correct_output):
    assert np.allclose(nearest_neighbor, correct_output)

@pytest.mark.parametrize("nn_bad_input", ["", [], 1.0])
def test_nn_bad_inputs(nn_bad_input):
    msg = ".* object cannot be interpreted as an integer"

    with pytest.raises(TypeError, match = msg): 
        GS.nearest_neighbor(nn_bad_input)
