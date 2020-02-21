from pybees.utils.validation import (
    check_input_array, 
    check_coordinate_array, 
    check_iterations,
    check_plot_history,
    check_discrete_func,
    check_continuous_func
)

from pybees.utils.continuous_single_obj import levy
from pybees.utils.combinatorial_single_obj import tour_distance
from pybees.bees_algorithm._simple_bees_algorithm import (
    SimpleBeesDiscrete,
    SimpleBeesContinuous
)

import numpy as np
import pytest

def test_check_input_array():

    # bad inputs
    # -------------------------------------------------------------------------

    with pytest.raises(TypeError) as e_info:
        check_input_array(np.array([[1,3,4], [1,2,3,4]]))

    assert str(e_info.value) == "Bad dtype('O'). Must contain either " \
        "ints or floats"

    with pytest.raises(TypeError) as e_info:
        check_input_array(np.array([[1,3,"",4], [1,2,3,4]]))

    assert str(e_info.value) == "Bad dtype('<U11'). Must contain either " \
        "ints or floats"

    with pytest.raises(ValueError) as e_info:
        check_input_array(np.array([[[1], [1]]]))

    assert str(e_info.value) == "Bad shape (1, 2, 1). Must have 1-2 dimensions"

    with pytest.raises(ValueError) as e_info:
        check_input_array(np.array([]))

    assert str(e_info.value) == "Bad shape (0,)"

    with pytest.raises(ValueError) as e_info:
        check_input_array(np.array([[1], [1]]), two_dim=True)
    
    assert str(e_info.value) == 'Bad shape (2, 1). ``x`` must have ' \
        'shape (n_coordinates, 2). Try shape (2, 2)'

    # Check input/ouput
    # -------------------------------------------------------------------------

    # Check for functions requiring arr.shape[1] == 2
    assert np.allclose(check_input_array(np.array([1])), np.array([[1]]))
    assert np.allclose(
        check_input_array(np.array([[1], [1]])), 
        np.array([[1], [1]])
    )
    assert np.allclose(
        check_input_array(np.array([[1,2,3], [1,2,3]])), 
        np.array([[1, 2, 3], [1, 2, 3]])
    )

    # Check for functions requiring arr.shape[1] < 0
    assert np.allclose(
        check_input_array(np.array([1, 2]), two_dim=True), 
        np.array([[1, 2]])
    )
    assert np.allclose(
        check_input_array(np.array([[1, 2], [1, 2]]), two_dim=True), 
        np.array([[1, 2], [1, 2]])
    )

def test_n_iterations():
    
    for i in range(1, 100):
        assert check_iterations(i) == i

    with pytest.raises(ValueError) as e_info:
        check_iterations(0)

    assert str(e_info.value) == '``n_iter`` must be greater than 0'

    invalid_types = [1., "", []]
    
    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            check_iterations(t)
        
        assert str(e_info.value) == '``n_iter`` must be of type ``int``'


def test_check_plot():

    # bad inputs
    # -------------------------------------------------------------------------
    bad_inputs = [np.array([]), "", []]

    for i in bad_inputs:
        with pytest.raises(TypeError) as e_info:
            check_plot_history(i)
        
        assert str(e_info.value) == '``optimiation_object`` must be a ' \
            f'subclass of BaseBeesAlgorithm. Detected {type(i)}'

    # Correct input 
    # -------------------------------------------------------------------------

    sbc = SimpleBeesContinuous(
        n_scout_bees = 20, 
        elite_site_params = (5,2), 
        best_site_params = (5, 1),
        bounds = (-10,10), 
        n_dim = 2,
        nbhd_radius = 1.5,
    )

    sbd = SimpleBeesDiscrete(
        n_scout_bees = 10, 
        elite_site_params = (1, 4), 
        best_site_params = (1, 3), 
        coordinates = np.random.randint(10, size=[10, 2])
    )

    msg = 'No data detected. Please execute self.optimize'
    with pytest.raises(AttributeError, match=msg) as e_info:
        sbc.plot()

    with pytest.raises(AttributeError, match=msg):
        sbd.plot()

def test_check_coordinate_array():

    # Incorrect input type 
    # -------------------------------------------------------------------------

    msg_one = '`coordinates` must be an np.ndarray. Detected .*'

    with pytest.raises(TypeError, match = msg_one):
        check_coordinate_array("")
        check_coordinate_array([])

    # Test for finite values 
    # -------------------------------------------------------------------------
    
    msg_two = "Input contains NaN, infinity or a value too large for " \
        r"dtype\('float64'\)."

    with pytest.raises(ValueError, match = msg_two):
        check_coordinate_array(np.array([[0, 1], [0, 1], [-np.nan, 1]]))
        check_coordinate_array(np.array([[0, 1], [0, 1], [np.nan, 1]]))

    # Incorrect element types
    # -------------------------------------------------------------------------

    msg_three = r"Detected incorrect type: dtype\(.*\). `coordinates` must " \
        "contain either integers/floats. Try, " \
        r"`your_array = your_array.astype\(np.float64\).`"

    with pytest.raises(TypeError, match = msg_three):
        check_coordinate_array(np.array([[0, 1], [0, 1], ["", 1]]))
        check_coordinate_array(np.array([[0, 1], [0, 1], ""]))

    # Incorrect input shape 
    # -------------------------------------------------------------------------

    msg_four = r'Bad shape .*. `coordinates` must have shape \(m, n\) where ' \
        '`m` is the number of coordinates and `n` is the number of ' \
        'dimensions. See the examples.'

    with pytest.raises(ValueError, match = msg_four):
        check_coordinate_array(np.random.randint(10, size=[3, 2, 1]))
        check_coordinate_array(np.random.randint(10, size=[3]))

    msg_five = r'Bad shape .*. .* dimension/s and .* coordinate/s were ' \
        'detected. `coordinates` must have at least 3 coordinates and 2 ' \
        'dimensions. `coordinates` could take following form which has 3 ' \
        'coordinates and 2 dimensions.\n\n.*'

    with pytest.raises(ValueError, match = msg_five):
        check_coordinate_array(np.random.randint(10, size=[1, 2]))
        check_coordinate_array(np.random.randint(10, size=[3, 1]))

    # Check correct input
    # -------------------------------------------------------------------------

    arr = np.random.randint(10, size=[10, 2])
    assert np.alltrue(check_coordinate_array(arr) == arr.astype(np.float64))

def test_check_continuous_func():
    params = dict(n_scout_bees = 10, n_dim=2)

    # Incorrect number of inputs  
    # -------------------------------------------------------------------------

    msg_one = '`func` should accept an np.ndarray with shape  ' \
        r'\(dimension, n\) where dimension >= 1 and n >= 1. `dimension` is the ' \
        'number of dimensions a coordinate has and n is the number of point  ' \
        r'coordinates. `func` should return an np.ndarray with shape \(m,\).See ' \
        r'the examples for SimpleBeesContinuous\(\)'

    with pytest.raises(AttributeError, match = msg_one):
        check_continuous_func(lambda x, y: None, **params)
        check_continuous_func(lambda : None, **params)

    # Incorrect output type
    # -------------------------------------------------------------------------

    msg_two = r'`func` return must be an np.ndarray. Detected .*'

    with pytest.raises(TypeError, match = msg_two):
        check_continuous_func(lambda x: None, **params)
        check_continuous_func(lambda x: 1, **params)

    # Incorrect output shape
    # -------------------------------------------------------------------------

    msg_three = r'Bad output shape \(10, 2\). `func` should return an array ' \
        r'with shape \(n, \) where n is the number of point coordinates. ' \
        'Please see the example functions. E.g. ' \
        r'func\(np.random.randint\(10, size = \[10, 5\]\)\) should return ' \
        r'shape \(10,\).'

    with pytest.raises(ValueError, match = msg_three):
        check_continuous_func(lambda x: x, **params)

