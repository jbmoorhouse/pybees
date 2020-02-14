"""
This module gathers local and global search operations for combinatorial
problems. These include but are not limited to

* tour_distance
"""

# Authors: Joseph Moorhouse <moorhouse@live.co.uk>
#
# License: BSD 3 clause


# =============================================================================
# Cost functions
# =============================================================================


def _tour_coordinates(coordinates):
    """Return an array representing a round trip tour.

    This functions appends the first index value to the input
    array. This functions allows the users to calculate the distance
    between each point, when considering a total round trip tour

    Parameters
    ----------
    coordinates : ndarray, shape(destinations, dimensions)
        The array containing the coordinates of the "destinations"

    Returns
    -------
    tour: ndarray, shape(destinations + 1, dimensions)
        The array, with the initial position concatenated to the input
        array. This allows the user to calculate the round trip tour
        distance
    """
    x = coordinates

    if x.ndim <= 2:
        x = x[np.newaxis, :]

    return np.array([np.concatenate((a, a[0, np.newaxis])) for a in x])

def tour_distance(bee_permutations, coordinates):
    """Calculate the round trip distance of the input array.

    Returns
    -------
    distance: float
        Round trip distance between the coordinates in the input array.
    """

    c = coordinates[bee_permutations]
    tour = _tour_coordinates(c)
    
    tour_dist = np.array(
        [np.linalg.norm(t[1:, :] - t[:-1, :], axis=1).sum() for t in tour])

    return tour_dist