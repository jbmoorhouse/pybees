from pybees.utils.combinatorial_search import GlobalSearch
from scipy.spatial.distance import cdist

def test_instance():
    np.random.seed(0)
    arr = np.random.randint(10, size=[10, 2])

    gs = GlobalSearch(arr)
    
    assert gs.coordinates == arr
    assert gs.distance == cdist(arr, arr)

