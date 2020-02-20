from pybees.utils.validation import *

import numpy as np
import pytest

def test_check_input_array():

    # bad input
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
