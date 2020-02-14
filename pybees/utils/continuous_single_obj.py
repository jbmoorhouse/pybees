"""
This file contains a collection of test functions for testing the Bees 
algorithm for optimisation problems
"""

# Authors: Joseph Moorhouse
#
# License: BSD 3 clause

import numpy as np 


# =============================================================================
# Types and constants
# =============================================================================

PI = np.pi

# =============================================================================
# Many local functions
# =============================================================================


def levy(x):
    """Implementation of the levy function. 
    
    Please see https://www.sfu.ca/~ssurjano/levy.html for more details. The 
    function is typically evaluated on the hypercube xi ∈ [-10, 10], for all 
    i = 1, …, d. The function is applied column-wise.
    
    Parameters
    ----------
    x : ndarray with shape (n_coordinates, n_dimensions)
        2D array of integers or floats. Each row represents the coordinates
        of a single point in a hypercube with n_dimensions.

    Returns
    -------
    z : ndarray with shape (n_dimensions,)
        The output from the levy function as defined. 

    Examples
    --------
    >>> x = np.random.randint(-10,10,size=[10,2])
    >>> x
    array([[ -6,  -7],
           [  1,   1],
           [  9,  -6],
           [-10,  -1],
           [ -2,   6],
           [ -7,   3],
           [  7,  -8],
           [  6,   6],
           [  1, -10],
           [ -2,   8]]))

    >>> levy(x)
    array([3.67986168e+01, 1.49975978e-32, 3.84479367e+01, 8.05078090e+01,
           9.55739901e+00, 3.25729367e+01, 1.99433481e+01, 2.01038861e+01,
           1.51250000e+01, 1.25573990e+01])
    """    
    x = x[np.newaxis, :] if x.ndim == 1 else x

    w = 1 + (x - 1) / 4

    term_one = np.sin(PI * w[:, 0]) ** 2
    term_three = ((w[:, -1]-1) ** 2) * (1+(np.sin(2*PI*w[:, -1])) ** 2)
    sum_ = np.sum(
        ((w[:, :-1]-1)**2) * (1+10*(np.sin(PI * w[:, :-1]+1))**2), 
        axis=1
    )

    return term_one + sum_ + term_three

