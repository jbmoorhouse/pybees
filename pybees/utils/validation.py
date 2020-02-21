"""
This module gathers a range of utility functions.
"""

# Authors: Joseph Moorhouse <moorhouse@live.co.uk>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import assert_all_finite
import inspect

from pybees.bees_algorithm._base import BaseBeesAlgorithm

__all__ = [
    'check_input_array', 
    'check_coordinate_array', 
    'check_iterations',
    'check_plot_history',
    'check_discrete_func',
    'check_continuous_func'
]

# =============================================================================
# Input checking
# =============================================================================

def check_input_array(x, two_dim=False):

    # Check type, element types and minimum size
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Bad {type(x)!r}. Must pass a np.ndarray")
    elif x.dtype.kind not in "fi":
        raise TypeError(f"Bad {x.dtype!r}. Must contain either ints or floats")
    elif x.size == 0:
        raise ValueError(f"Bad shape {x.shape}")

    # x must have either 1 or 2 dimensions. If 1, then reshape.
    if x.ndim > 2:
        raise ValueError(f"Bad shape {x.shape}. Must have 1-2 dimensions")
    elif x.ndim == 1:
        x = x[np.newaxis, :] 

    # If the function check_input_array is passed to requires 2 dimensions,
    # Ensure that the input array has the correct shape
    if two_dim:
        if x.shape[1] != 2:
            raise ValueError(f"Bad shape {x.shape}. ``x`` must have "
                            "shape (n_coordinates, 2). Try "
                            f"shape ({x.shape[0]}, 2)")

    return x

def check_coordinate_array(coordinates):
    """Input validation on np.ndarray for combinatorial optimization problems.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure [1]. The input is also checked 
    against the minimum shape requirement of (3, 2)  

    Parameters
    ----------
    coordinates : ndarray
        Input object to check / convert.
        
    Returns
    -------
    coordinates_converted : np.ndarray
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.utils.
    check_array.html
    
    Examples
    --------
    Checking a valid array.
    
    >>> coordinates = np.array([[14, 17], [15, 11], [10,  3]])
    >>> converted_coordinates = check_coordinate_array(coordinates) 
    ...
    array([[14., 17.], 
           [15., 11.], 
           [10.,  3.]])
    
    
    
    Checking an array containing a string.
    
    >>> coordinates np.array([[1,2,3], [1,2,"3"], [1,2,3]])
    >>> converted_coordinates = check_coordinate_array(coordinates) 
    ...
    ValueError: Detected incorrect type: dtype('<U11'). `coordinates` 
        must contain either integers/floats. Try,
        `your_array = your_array.astype(np.float64).`
    """

    example = np.array([[0,1], [2,3], [4,5]])
    
    if not isinstance(coordinates, np.ndarray):
        raise TypeError(f"`coordinates` must be an np.ndarray. Detected "
                        f"{coordinates!r}")
        
    # Use sklearn.utils.validation.assert_all_finite to determine if all 
    # inputs to 'coordinates' are finite
    assert_all_finite(coordinates)
    
    if coordinates.dtype.kind not in "fi":
        raise TypeError(f"Detected incorrect type: {coordinates.dtype!r}. "
                         "`coordinates` must contain either integers/floats. "
                         "Try, `your_array = your_array.astype(np.float64).`")    
    elif coordinates.ndim != 2:
        raise ValueError(f"Bad shape {coordinates.shape}. `coordinates` must " 
                         "have shape (m, n) where `m` is the number of "
                         "coordinates and `n` is the number of dimensions. "
                         "See the examples.")
        
    m, n = coordinates.shape
    
    if n < 2 or m < 3: 
        raise ValueError(f"Bad shape {(m, n)}. {n} dimension/s and {m} "
                         f"coordinate/s were detected. `coordinates` must have"
                         " at least 3 coordinates and 2 dimensions. "
                         "`coordinates` could take following form which has 3"
                         f" coordinates and 2 dimensions.\n\n{example!r}")
    
    return coordinates.astype(np.float64)

def check_iterations(n_iter):
    """Check n_iter input"""

    if not isinstance(n_iter, int):
        raise TypeError('``n_iter`` must be of type ``int``')
    elif n_iter < 1:
        raise ValueError('``n_iter`` must be greater than 0')

    return n_iter

def check_plot_history(optimiation_object):
    """Check the optimisation object before plotting"""

    if not issubclass(type(optimiation_object), BaseBeesAlgorithm):
        raise TypeError("``optimiation_object`` must be a subclass of "
                        "BaseBeesAlgorithm. Detected "
                        f"{type(optimiation_object)}")
    elif not hasattr(optimiation_object, "history"):
        raise AttributeError("No data detected. Please execute self.optimize")

# =============================================================================
# User defined function checking
# =============================================================================

def check_discrete_func(func, size):
    """Combinatorial cost function validation function
    
    The input function is checked with a valid input and it's output is 
    assessed for correctness. Specifically, the cost function should accept
    a `permutations` arguments, which represents an np.ndarray with shape
    (n_permutations, n_coordinates). This array is used to permute the 
    coordinates also passed to the function. With the coordinates permuted
    for `n_permutations`, the "cost" for each is assessed. For example, a 
    typical cost function is the total tour distance, which for a given 
    permutation, calculates the distance between all the points in a round 
    trip.
    
    Parameters
    ----------
    func : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    
    Raises
    ------
    AttributeError
        [description]
    TypeError
        [description]
    IndexError
        [description] 
    """
    try:
        arr = np.random.randint(0, 10, [10, 2])
        permutations = np.random.rand(size, arr.shape[0]).argsort(1)
        
        cost = func(permutations, arr)
    except:
        raise AttributeError('``func`` should accept 2 parameters. ' 
            '``bee_permutations`` should be an np.ndarray with shape ' 
            '``(n_permutations, n_coordinates)``, which represents ' 
            '``n_permutations`` of some ``range(coordinates)``. For example '
            '``np.array([0,1,2], [2,1,0])`` where ``n_permutations = 2``. '
            'The second parameter, ``coordinates``, is an np.ndarray with '
            'shape ``(n_coordinates, n_dimensions)``. ``func`` should return '
            'an np.ndarray with shape ``(n_permutations,)``. Please see '
            '``combinatorial_single_obj.py`` for examples.')

    if not isinstance(cost, np.ndarray):
        raise TypeError("``cost_function`` should return an np.ndarray")
    elif cost.dtype.kind not in "fi":
        raise TypeError("``cost_function`` should return an np.ndarray with "
            "elements of type int or float")
    elif cost.ndim != 1 or cost.size != size:
        raise ValueError(f"Bad shape {cost.shape}. func should return "
            "np.ndarray with shape (n_permutations,).")


def check_continuous_func(func, n_scout_bees, n_dim):
    """[summary]
    
    Parameters
    ----------
    func : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    
    Raises
    ------
    AttributeError
        [description]
    TypeError
        [description]
    IndexError
        [description]
    """

    try:
        x = np.random.randint(10, size = [n_scout_bees, n_dim])
        res = func(x)
    except:
        raise AttributeError("`func` should accept an np.ndarray with shape  "
                             "(dimension, n) where dimension >= 1 and n >= 1. " 
                             "`dimension` is the number of dimensions a "
                             "coordinate has and n is the number of point "
                             " coordinates. `func` should return an "
                             "np.ndarray with shape (m,).See the examples "
                             "for SimpleBeesContinuous()")
        
    if not isinstance(res, np.ndarray):
        raise TypeError(f"`func` return must be an np.ndarray. Detected {res}")
        
    elif res.ndim != 1 or res.size != x.shape[0]:
        raise ValueError(f"Bad output shape {res.shape}. `func` should return " 
                         "an array with shape (n, ) where n is the number of point "
                         "coordinates. Please see the example functions. E.g. "
                         "func(np.random.randint(10, size = [10, 5])) should "
                         "return shape (10,).")


