from pybees.utils.combinatorial_single_obj import tour_distance
import numpy as np
import pytest

# =============================================================================
# tour_distance
# =============================================================================

@pytest.mark.parametrize(
    "bee_permutations, coordinates",
    [(np.random.rand(2, 3).argsort(1), 
      np.random.randint(10, size = [10, 2])),
     (np.random.rand(2, 10).argsort(1), 
      np.random.randint(10, size = [3, 2]))]
)
def test_tour_bad_inputs(bee_permutations, coordinates):
    msg = r"Bad shape. Must satisfy bee_permutations\[1\] == " \
        "coordinates.shape\[0\]. Detected bee_permutation.shape\[1\] = .* " \
        "and coordinates.shape\[0\] = .*"

    with pytest.raises(ValueError, match = msg):
        tour_distance(bee_permutations, coordinates)

np.random.seed(0)

@pytest.mark.parametrize(
    "bee_permutations, coordinates, solution",
    [(np.random.rand(10, 10).argsort(1), 
      np.random.randint(10, size = [10, 2]),
      np.array([55.79739851, 47.83300624, 51.94952122, 61.85755991, 
                62.47639945, 57.58530909, 67.43100982, 62.36886206, 
                46.49171438, 55.54946681])),
     (np.random.rand(10, 20).argsort(1), 
      np.random.randint(10, size = [20, 2]),
      np.array([ 94.2668111 ,  94.53465832,  89.86530473,  90.83521105,
                 96.58170752,  84.29433697,  99.41701685,  99.14881104,
                108.80793023,  98.07230858]))]
)
def test_tour_good_inputs(bee_permutations, coordinates, solution):
    res = tour_distance(bee_permutations, coordinates)

    assert np.allclose(res, solution)