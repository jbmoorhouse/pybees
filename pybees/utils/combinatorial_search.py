"""
This module gathers local and global search operations for combinatorial
problems. These include but are not limited to

local search methods
--------------------
* swap
* reversion
* insertion

global search methods
---------------------
* permute
* nearest_neighbour
"""

# Authors: Joseph Moorhouse <moorhouse@live.co.uk>
#
# License: BSD 3 clause

import numpy as np
from scipy.spatial.distance import cdist

__all__ = [
    "swap",
    "reversion",
    "insertion",
    "GlobalSearch",
    "tour_distance"
]

# =============================================================================
# Search utility functions
# =============================================================================

def _random_choice_unique(arr):
    """Generate random sample from 2D array with replacement along row axis
    
    This function accepts a 2D array and returns unique indices for each row,
    but with the possibility of repeated values over multiple rows [1]. This 
    simulates the behaviour of `np.random.choice` when replace is set to `False`
    on a 1D array, but over multiple rows
    
    Parameters
    ----------
    arr: np.ndarray with shape(m, n)
        2D array, for which random indices will be generated for each row. 
    
    Returns
    -------
    indices: np.ndarray with shape(m, 2)
        2D array with 2 random indices. Values can be repeated from row to row, 
        but each row contains unique values
    
    References 
    ----------
    [1] https://stackoverflow.com/questions/51279464/sampling-unique-column
    -indexes-for-each-row-of-a-numpy-array.
    
    Examples
    --------
    >>> X = np.repeat(np.arange(5)[np.newaxis], 2, 0)
    >>> X
    ...
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])
    ...
    >>> random_choice_unique(X)
    ...
    array([[0, 4],
           [0, 3],
           [2, 3]])
    
    >>> X = np.repeat(np.arange(2)[np.newaxis], 2, 0)
    >>> X
    ...
    array([[0, 1],
           [0, 1],
           [0, 1]])
    ...
    >>> random_choice_unique(X)
    ...
    IndexError: It is advised that `arr` has 3 or more columns
    """
    if arr.ndim != 2:
        raise IndexError("`arr` must have shape (m, n)")

    # Check is `arr` has 3 or more columns. 3 or more columns is enforced since 
    # any less than this will result in identical and predicatable results
    m, n = arr.shape
    
    if (m < 1 or n < 3):
        raise IndexError("`arr` has {} rows and {} columns. It is advised that "
                         "`arr` has 3 or more columns and 1 or more rows."
                         .format(m, n))
    
    # Get sorted random indices to slice 2D array before mutation.
    return np.sort(np.random.rand(m,n).argsort(1)[:,:2], axis=1)
    
def _prepare_array(bee_permutation, n_bees):

    if bee_permutation.ndim != 1:
        raise IndexError("Bad shape {}. `bee_permutation must have shape "
                         "(n_coordinates, ).")

    # Create copy of bee_permuation and initialise the forager bees
    temp = bee_permutation.copy()[np.newaxis]
    forager_bees = np.repeat(temp, n_bees, axis=0)

    # Generate the random coordinates to swap
    return forager_bees, _random_choice_unique(forager_bees)

# =============================================================================
# Local search methods
# =============================================================================

def swap(bee_permutation, n_bees):
    """Foraging stage using the swap mutation method.

    This function simulates the foraging stage of the algorithm. It takes the
    current bee permutation of a single bee and mutates the order using a swap 
    mutation step. `n_bees` forager bees are created by swapping two unique 
    indices per row.

    Parameters
    ----------
    bee_permutation: np.ndarray with shape (1, n_coordinates) 
        Array representing the indexing permutation of the discrete bee
        coordinates. 

    Returns
    -------
    forager_bees: np.ndarray with shape (n_bees, n)
        The new indexing permutations, using the swap mutation approach.

    Examples
    --------
    >>> bee_permutation = np.arange(10)[np.newaxis, :]
    >>> bee_permutation
    array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    ...
    >>> swap(bee_permutation, 2)
    array([[6, 1, 2, 3, 4, 5, 0, 7, 8, 9],
           [0, 1, 9, 3, 4, 5, 6, 7, 8, 2]])

    """

    # Prepare the forager bees and mutation indices
    forager_bees, swap = _prepare_array(bee_permutation, n_bees)

    # Mutate the forager bees. `h` is helper array to coerces forage into the 
    # correct shape.  Question asked here for this procedure https://
    # stackoverflow.com/questions/59936869/swap-multiple-indices-in-2d-array
    h = np.arange(n_bees)[:, np.newaxis]
    forager_bees[h, swap] = forager_bees[h, swap[:, ::-1]]  

    return forager_bees.astype(int)


def reversion(bee_permutation, n_bees):
    """Foraging stage using the reverse mutation method.

    This function simulates the foraging stage of the algorithm. It takes the
    current bee permutation of a single bee and mutates the order using a 
    reversion mutation step. Unique `start` and `stop` positions are randomly
    generated for `n_bees`. These `start` and `stop` positions are used to 
    reverse the order between these indices.

    Parameters
    ----------
    bee_permutation: np.ndarray with shape (1, n_coordinates) 
        Array representing the indexing permutation of the discrete bee
        coordinates. 

    Returns
    -------
    forager_bees: np.ndarray with shape (n_bees, n)
        The new indexing permutations, using the reversion mutation approach.


    Examples
    --------
    >>> bee_permutation = np.arange(10)[np.newaxis, :]
    >>> bee_permutation
    ...
    array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    ...
    ...
    >>> reversion(bee_permutation, 2)
    ...
    array([[2, 1, 0, 3, 4, 5, 6, 7, 8],
           [0, 3, 2, 1, 4, 5, 6, 7, 8]])
    """

    # Prepare the forager bees and mutation indices
    forager_bees, reverse = _prepare_array(bee_permutation, n_bees)

    # Mutate the original array.
    for idx, (i, j) in enumerate(reverse):
        forager_bees[idx, i:j+1] = forager_bees[idx, i:j+1][::-1]

    return forager_bees.astype(int)


def insertion(bee_permutation, n_bees):
    """Foraging stage using the insertion mutation method.

    This function simulates the foraging stage of the algorithm. It takes the
    current bee permutation of a single bee and mutates the order using an 
    insetion mutation step. Unique `start` and `stop` positions are randomly
    generated for `n_bees`. The array is subsequently rolled. Elements that 
    roll beyond the `stop` position are re-introduced at the `start` 

    Parameters
    ----------
    bee_permutation: np.ndarray with shape (1, n_coordinates)
        Array representing the indexing permutation of the discrete bee
        coordinates.

    Returns
    -------
    forager_bees: np.ndarray with shape (n_bees, n)
        The new indexing permutations, using the insertion mutation approach.


    Examples
    --------
    >>> bee_permutation = np.arange(10)[np.newaxis, :]
    >>> bee_permutation
    ...
    array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    ...
    ...
    >>> reversion(bee_permutation, 2)
    ...
    array([[0, 1, 6, 2, 3, 4, 5, 7, 8, 9],
           [0, 3, 1, 2, 4, 5, 6, 7, 8, 9]])
    """

    # Prepare the forager bees and mutation indices
    forager_bees, insert = _prepare_array(bee_permutation, n_bees)

    for idx, (i, j) in enumerate(insert):
        forager_bees[idx, i:j + 1] = np.roll(
            forager_bees[idx, i:j + 1], 1, axis=0)

    return forager_bees.astype(int)

# =============================================================================
# Global search methods
# =============================================================================

class GlobalSearch:
    
    def __init__(self, coordinates):
        
        self.coordinates = coordinates
        self.distance = cdist(coordinates, coordinates)

    def permute(self, n_bees):
        """Global search method

        This global search method generates `n_bees` random permuations to 
        mutate a 2D array of coordinates with shape (2, n), where n is the 
        number of discrete coordinates.

        Parameters
        ----------    
        n_bees : int
            The total number of permutations to generate.

        Returns
        -------
        p: np.ndarray with shape (n_bees, n_coordinates)
            `n_bees` random permuations to mutate a 2D coordinate matrix

        Examples
        --------
        >>> coordinates = np.array([[0,1], [3,2], [9,4], [1,7]])
        >>> coordinates
        ...
        array([[0, 1],
               [3, 2],
               [9, 4],
               [1, 7]])
        ...
        >>> p = permute(coordinates.shape[0], 2)
        >>> p
        ...
        array([[0, 3, 1, 2],
               [1, 0, 3, 2]]))
        ...
        >>> coordinates[p]
        ...
        array([[[0, 1],
                [1, 7],
                [3, 2],
                [9, 4]],

               [[3, 2],
                [0, 1],
                [1, 7],
                [9, 4]]])
        """
        return np.random.rand(n_bees, self.coordinates.shape[0]).argsort(1)

    def nearest_neighbor(self, n_bees):
        n_coords = self.coordinates.shape[0]

        def helper(n_coords):    
            idx = np.arange(n_coords)
            path = np.empty(n_coords, dtype=int)
            mask = np.ones(n_coords, dtype=bool)

            last_idx = np.random.randint(n_coords)
            path[0] = last_idx
            mask[last_idx] = False

            for i in range(1, n_coords):
                last_idx = idx[mask][np.argmin(self.distance[last_idx, mask])]
                path[i] = last_idx
                mask[last_idx] = False

            return path

        return np.array([helper(n_coords) for _ in range(n_bees)]).astype(int)
