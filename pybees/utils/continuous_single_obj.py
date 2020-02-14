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
# Many local minima functions
# =============================================================================

def levy(x):
    """Implementation of the levy function. 
    
    Please see https://www.sfu.ca/~ssurjano/levy.html for more details. The 
    function is typically evaluated on the hypercube xi ∈ [-10, 10], for all 
    i = 1, …, d. The function is applied column-wise.
    
    Parameters
    ----------
    x : ndarray with shape (n_coordinates, n_dimensions)
        1D or 2D array of integers or floats. Each row represents the 
        coordinates of a single point in a hypercube with n_dimensions.

    Returns
    -------
    res : ndarray with shape (n_coordinates,)
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


def ackley(x, a = 20, b = 0.2, c = 2 * PI):
    """Implementation of the ackley function. 

    Please see https://www.sfu.ca/~ssurjano/ackley.html for more details.
    Recommended variable values are: a = 20, b = 0.2 and c = 2π. The function 
    is usually evaluated on the hypercube xi ∈ [-32.768, 32.768], for all 
    i = 1, …, d, although it may be restricted to a smaller domain. The 
    function is applied column-wise

    Parameters
    ----------
    x : ndarray with shape (n_coordinates, n_dimensions)
        1D or 2D array of integers or floats. Each row represents the 
        coordinates of a single point in a hypercube with n_dimensions.
    a : int or float, default: 20
        function constant, see link for more details
    b : int or float, default: 0.2
        function constant, see link for more details
    c : int or float, default: 2 * np.pi
        function constant, see link for more details

    Returns
    -------
    res : ndarray with shape (n_coordinates,)
        The output from the ackley function as defined.

    Examples
    --------
    >>> x = np.random.randint(-10,10,size=[10,2])
    >>> x
    array([[ -7,   4],
           [ -2,   8],
           [ -1,   5],
           [  7,  -7],
           [  4,   5],
           [  5,  -1],
           [ -3,  -2],
           [  5,   8],
           [ -7,   0],
           [ -5, -10]]))

    >>> ackley(x)
    array([13.60474155, 13.76896846, 10.27575727, 15.06806072, 
           11.91351815, 10.27575727,  7.98891081, 14.73244732, 
           12.56809083, 15.88518678])
    """

    x = x[np.newaxis, :] if x.ndim == 1 else x
    d = x.shape[1]
        
    sum_one = np.sum(x ** 2, axis=1)
    sum_two = np.sum(np.cos(c * x), axis=1)
     
    term_one = -a * np.exp(-b * ((sum_one/d) ** 0.5))
    term_two = -np.exp(sum_two/d)

    return term_one + term_two + a + np.exp(1)

def drop_wave(x):
    """Implementation of the drop wave function. 
    
    Please see https://www.sfu.ca/~ssurjano/drop.html for more details. The 
    function is usually evaluated on the square xi ∈ [-5.12, 5.12], for 
    all i = 1, 2. The function is applied column-wise

    Parameters
    ----------
    x : ndarray with shape (n_coordinates, 2)
        1D or 2D array of integers or floats. Each row represents the 
        coordinates of a single point in a hypercube with 2 dimensions. 
        
    Returns
    -------
    res : ndarray with shape (n_coordinates,)
        the output from the drop_wave function as defined.

    Examples
    --------
    >>> x = np.random.randint(-10,10, size=[10,2])
    >>> x
    array([[ -7,   1],
           [ -7,   9],
           [  1, -10],
           [  2,  -8],
           [ -2,  -7],
           [ -7,  -6],
           [ -2,   5],
           [  8,   6],
           [ -4,   0],
           [ -2,   7]]))

    >>> drop_wave(x)
    array([-1.64573188e-05, -1.73293421e-02, -2.56292536e-02, 
           -2.76212905e-02, -6.39818116e-02, -4.98128978e-03,   
           -4.74197545e-02, -3.48880956e-02, -3.59855661e-02, 
           -6.39818116e-02])
    """
    x = x[np.newaxis, :] if x.ndim == 1 else x          
    
    if x.shape[1] != 2:
        raise ValueError(f"Bad shape {x.shape}. ``x`` must have "
                         "shape(n_coordinates, 2). Try "
                         f"shape({x.shape[0]}, 2)")
        
    frac_one = 1 + np.cos(12 * ((x[:,0] ** 2 + x[:,1] ** 2) ** 0.5))
    frac_two = 0.5 * (x[:,0] ** 2 + x[:,1] ** 2) + 2

    return -frac_one / frac_two


# =============================================================================
# Steep ridges/drops
# =============================================================================

def de_jong(x):
    """Implementation of the de-jong 5th function. 
    
    Please see https://www.sfu.ca/~ssurjano/dejong5.html for more details.

    Parameters
    ----------
    x : ndarray with shape (n_coordinates, 2)
        1D or 2D array of integers or floats. Each row represents the 
        coordinates of a single point in a hypercube with 2 dimensions.
        
    Returns
    -------
    res : ndarray with shape (n_coordinates,)
        the output from the drop_wave function as defined.

    Examples
    --------
    >>> x = np.random.randint(-10,10, size=[10,2])
    >>> x
    array([[ 7, -4],
           [ 3,  4],
           [ 7, -4],
           [ 9,  7],
           [-7,  8],
           [-8, -1],
           [ 9, -1],
           [ 5, -7],
           [ 4, -3],
           [-9,  5]]))

    >>> de_jong(x)
    array([497.32791193, 453.01047752, 497.32791193, 497.92766864,
           498.05151155, 498.03139838, 497.34461417, 497.42674123,
           453.01047735, 497.42359525])
    """

    x = x[np.newaxis, :] if x.ndim == 1 else x 
    
    if x.shape[1] != 2:
        raise ValueError(f"Bad shape {x.shape}. ``x`` must have "
                         "shape(n_coordinates, 2). Try "
                         f"shape({x.shape[0]}, 2)")
        
    a = np.array([-32,-16,0,16,32])
    a_one = np.repeat(np.tile(a, 5)[np.newaxis, :], x.shape[0], axis=0)
    a_two = np.repeat(np.repeat(a, 5)[np.newaxis, :], x.shape[0], axis=0)
    
    i = np.repeat(np.arange(1, 26)[np.newaxis,:], x.shape[0], axis=0)
    
    sum_denom = (
        (i + (x[:,0][:, np.newaxis] - a_one) ** 6 + 
        (x[:,1][:, np.newaxis] - a_two) ** 6))

    return 1 / (0.002 + np.sum(1 / sum_denom, axis = 1)) 
    

