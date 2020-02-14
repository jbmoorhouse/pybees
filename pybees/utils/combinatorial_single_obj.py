"""
This module gathers the a collection of cost functions for combinatorial
optimization problems. These include but are not limited to

* tour_distance
"""

# Authors: Joseph Moorhouse <moorhouse@live.co.uk>
#
# License: BSD 3 clause

import numpy as np

# =============================================================================
# Cost functions
# =============================================================================


def _tour_coordinates(coordinates):
    """Concatenate the first coordinate to after the last index."""
    
    x = coordinates

    if x.ndim <= 2:
        x = x[np.newaxis, :]

    return np.array([np.concatenate((a, a[0, np.newaxis])) for a in x])

def tour_distance(bee_permutations, coordinates):
    """Calculate the round trip distance of the input array.

    Parameters
    ----------
    bee_permutations: ndarray with shape(n_permutations, n_coordinates)

    coordinates : ndarray, shape(n_coordinates, n_dimensions)
        The array containing the coordinates of the "destinations"

    Returns
    -------
    distances: ndarray with shape(n_permuatations,)
        Round trip distance between the coordinates in the input array.

    Examples
    --------
    >>> coordinates = np.random.randint(0, 50, [15, 2])
    >>> coordinates
    array([[44 47]
           [ 0  3]
           [ 3 39]
           [ 9 19]
           [21 36]
           [23  6]
           [24 24]
           [12  1]
           [38 39]
           [23 46]])

    >>> bee_permutations = np.random.rand(2, 15).argsort(1)
    >>> bee_permuations
    np.array([[4 2 3 0 6 9 5 7 1 8]
              [2 4 8 7 0 6 3 9 1 5]])

    >>> tour_distance(bee_permuations coordinates)
    np.array([270.30924397 324.84195812])
    """

    c = coordinates[bee_permutations]
    tour = _tour_coordinates(c)
    
    tour_dist = np.array(
        [np.linalg.norm(t[1:, :] - t[:-1, :], axis=1).sum() for t in tour])

    return tour_dist



np.random.seed(0)

coordinates = np.random.randint(0, 50, [10, 2])
permutations = np.random.rand(2, 10).argsort(1)

print(coordinates)
print(permutations)
print(tour_distance(permutations, coordinates))