from pybees.utils.continuous_single_obj import *

import pytest
import numpy as np

# =============================================================================
# Test arrays
# =============================================================================

# n-dimensions
# -----------------------------------------------------------------------------

ARRAY_ONE = np.arange(10)
ARRAY_TWO = np.arange(100).reshape(-1, 10)

# 2-dimensions
# -----------------------------------------------------------------------------

ARRAY_THREE = np.arange(2)
ARRAY_FOUR = np.arange(20).reshape(-1, 2)

# =============================================================================
# Test solutions
# =============================================================================

# levy
# -----------------------------------------------------------------------------

levy_one = [44.87676901]
levy_two = [   44.87676901,   681.99177324,  1676.30859533,  4139.22212528,
             5830.41606493, 10573.77683403, 12507.19917783, 19985.65589948,
            21706.65793403, 32374.85932164]

# ackley
# -----------------------------------------------------------------------------

ackley_one = [13.12408691]
ackley_two = [13.12408691, 18.95983103, 19.85598334, 19.98031975, 19.99732228,
              19.99963638, 19.99995067, 19.99999331, 19.99999909, 19.99999988]


# drop_wave
# -----------------------------------------------------------------------------

drop_wave_one = [-7.37541583e-01]
drop_wave_two = [-7.37541583e-01, -2.06428945e-01, -5.02733445e-02, 
                 -4.98128978e-03, -2.68442982e-02, -1.96572296e-03, 
                 -7.83437933e-03, -6.51938912e-03, -5.20379883e-04, 
                 -5.79397234e-03]

# de_jong
# -----------------------------------------------------------------------------

de_jong_one = [13.61860666]
de_jong_two = [ 13.61860666, 308.52439368, 487.36435393, 497.65866108,
               498.04695412, 495.53691355, 453.06428025,   71.9156152,
                19.23062993, 309.3996521 ]


# easom
# -----------------------------------------------------------------------------

easom_one = [-2.84751216e-07]
easom_two = [-2.84751216e-007, -1.09694364e-001,  2.80681393e-003,
             -7.01113101e-011, -9.24507120e-027,  2.09834870e-050,
             -3.94551640e-077,  5.49217211e-114, -1.60886253e-156,
             -5.17987727e-206]

# michalewicz
# -----------------------------------------------------------------------------

michalewicz_one = [0.0938903]
michalewicz_two = [ 0.0938903 ,  1.31798268, -1.29747026, -0.15998275,  
                   0.08200963,  -1.02164574,  0.15208927, -1.23877412, 
                   1.41206148, -0.19619113]


# =============================================================================
# Utility functions
# =============================================================================

def input_test(func, two_dims=False):

    # bad type
    # -------------------------------------------------------------------------

    with pytest.raises(TypeError) as e_info:
        func("")

    assert str(e_info.value) == f"Bad {type('')}. Must pass a np.ndarray"

    # bad element type
    # -------------------------------------------------------------------------

    with pytest.raises(TypeError) as e_info:
        func(np.array([""]))

    assert str(e_info.value) == f"Bad dtype('<U1'). Must contain either ints " \
        "or floats"

    # bad shape
    # -------------------------------------------------------------------------

    with pytest.raises(ValueError) as e_info:
        func(np.array([]))

    assert str(e_info.value) == f"Bad shape (0,)"

    # bad input shape when only 2 dimensions accepted
    # -------------------------------------------------------------------------

    if two_dims:

        for arr in [np.array([1]), np.array([1,2,3]), np.array([[1], [1]])]:
            with pytest.raises(ValueError) as e_info:
                func(arr)

            arr = arr[np.newaxis, :] if arr.ndim == 1 else arr
            
            assert str(e_info.value) == f'Bad shape {arr.shape}. ``x`` must ' \
                f'have shape (n_coordinates, 2). Try shape ({arr.shape[0]}, 2)'

# =============================================================================
# Tests many local minima functions
# =============================================================================

def test_levy():
    "Test for levy cost function"
    
    assert np.allclose(levy(ARRAY_ONE), levy_one)
    assert np.allclose(levy(ARRAY_TWO), levy_two)

    input_test(levy)

def test_ackley():
    "Test for ackley cost function"
    
    assert np.allclose(ackley(ARRAY_ONE), ackley_one)
    assert np.allclose(ackley(ARRAY_TWO), ackley_two)

    input_test(ackley)

def test_drop_wave():
    "Test for drop_wave cost function"
    
    assert np.allclose(drop_wave(ARRAY_THREE), drop_wave_one)
    assert np.allclose(drop_wave(ARRAY_FOUR), drop_wave_two)

    input_test(drop_wave, two_dims=True)

# =============================================================================
# Tests plate shaped functions
# =============================================================================

def test_de_jong():
    "Test for de_jong cost function"
    
    assert np.allclose(de_jong(ARRAY_THREE), de_jong_one)
    assert np.allclose(de_jong(ARRAY_FOUR), de_jong_two)

    input_test(de_jong, two_dims=True)

def test_easom():
    "Test for easom cost function"
    
    assert np.allclose(easom(ARRAY_THREE), easom_one)
    assert np.allclose(easom(ARRAY_FOUR), easom_two)

    input_test(easom, two_dims=True)

def test_michalewicz():
    "Test for michalewicz cost function"
    
    assert np.allclose(michalewicz(ARRAY_ONE), michalewicz_one)
    assert np.allclose(michalewicz(ARRAY_TWO), michalewicz_two)

    input_test(michalewicz)