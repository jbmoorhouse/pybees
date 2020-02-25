from pybees.utils.combinatorial_single_obj import tour_distance
import numpy
import pytest

# =============================================================================
# Cost functions
# =============================================================================

@pytest.mark.parametrize(
    "bee_permutations, coordinates"
    [(np.random.rand(2, 3).argsort(1), 
      np.random.randint(10, size = [10, 2])),
     (np.random.rand(2, 10).argsort(1), 
      np.random.randint(10, size = [3, 2]))]
)
def test_bad_inputs(bee_permutations, coordinates):
    msg = r"Bad shape. Must satisfy bee_permutations[1] == " \
        "coordinates.shape[0]. Detected .* and .*"

    with pytest.raises(ValueError, match = msg):
        tour_distance(bee_permutations, coordinates)