from pybees.utils.validation import *

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

    with pytest.raises(TypeError) as e_info:
        check_input_array(np.array([[[1], [1]]]))

    assert str(e_info.value) == "Bad shape(1, 2, 1). Must have 1-2 dimensions"

    with pytest.raises(TypeError) as e_info:
        check_input_array(np.array([]))

    assert str(e_info.value) == "Bad shape(0,)"