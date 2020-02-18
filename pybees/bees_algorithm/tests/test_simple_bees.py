from pybees.utils.continuous_single_obj import (
    levy, 
    ackley, 
    drop_wave, 
    michalewicz
)

from pybees.utils.combinatorial_single_obj import tour_distance
from pybees.bees_algorithm._simple_bees_algorithm import (
    SimpleBeesContinuous, 
    SimpleBeesDiscrete
)

import pytest
import numpy as np

# =============================================================================
#  SimpleBeesContinuous tests
# =============================================================================

def check_output(sbc_obj, fun, cost, coords):
    res = sbc_obj.optimize(fun, 300)

    assert(np.allclose([res.fun], [cost]))
    assert(np.allclose([res.x], coords))


def test_simple():
    # test levy function with 2 dimensions and 4 dimensions
    # -------------------------------------------------------------------------

    sbc_levy_2 = SimpleBeesContinuous(
        n_scout_bees = 60, 
        elite_site_params = (20, 40), 
        best_site_params = (20, 30),
        bounds = (-10,10), 
        n_dim = 2,
        nbhd_radius = 1,
    )

    sbc_levy_4 = SimpleBeesContinuous(
        n_scout_bees = 100, 
        elite_site_params = (40, 100), 
        best_site_params = (40, 80),
        bounds = (-10,10), 
        n_dim = 4,
        nbhd_radius = 1.5,
    )

    check_output(sbc_levy_2, levy, 0, [1, 1])
    check_output(sbc_levy_4, levy, 0, [1,1,1,1])

    # test drop_wave, michalewicz function with 2 dimensions only
    # -------------------------------------------------------------------------

    sbc_drop_wave = SimpleBeesContinuous(
        n_scout_bees = 30, 
        elite_site_params = (10, 40), 
        best_site_params = (10, 30),
        bounds = (-5.12,5.12), 
        n_dim = 2,
        nbhd_radius = 1
    )

    sbc_michalewicz_2 = SimpleBeesContinuous(
        n_scout_bees = 20, 
        elite_site_params = (8, 30), 
        best_site_params = (6, 20),
        bounds = (0, np.pi), 
        n_dim = 2,
        nbhd_radius = 1,
    )
    
    check_output(sbc_drop_wave, drop_wave , -1, [0, 0])
    check_output(sbc_michalewicz_2, michalewicz, -1.8013, [2.20, 1.57])

    # test ackley function with 7 dimensions
    # -------------------------------------------------------------------------
    
    sbc_ackley_7 = SimpleBeesContinuous(
        n_scout_bees = 100, 
        elite_site_params = (40,100), 
        best_site_params = (30,60),
        bounds = (-32.768, 32.768),
        n_dim = 7,
        nbhd_radius = 7
    )

    check_output(sbc_ackley_7, ackley, 0, [0, 0])


def test_n_scout_bees():
    params = dict(
        elite_site_params = (40, 100), 
        best_site_params = (40, 80),
        bounds = (-10,10), 
        n_dim = 2,
        nbhd_radius = 1.5
    )

    # test with invalid integer inputs. 
    # -------------------------------------------------------------------------
    invalid_integers = [0, -1, -10]

    for v in invalid_integers:
        with pytest.raises(ValueError) as e_info:
            SimpleBeesContinuous(n_scout_bees = v, **params)
        
        assert str(e_info.value) == (
            f'Detected {v} scout bees. ``n_scout_bees`` must be > 2')



    # test with incorrect types
    # -------------------------------------------------------------------------
    invalid_types = [100.0, "", []]

    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(n_scout_bees = t, **params)

        assert str(e_info.value) == (
            '`n_scout_bees` must be of type `int`')



def check_site_value(param_name, params):
    # Check with too few elite sites
    params[param_name] = (0, 20)

    with pytest.raises(ValueError) as e_info:
            SimpleBeesContinuous(**params)

    assert str(e_info.value) == f'Detected 0 in {param_name}. All ' \
        'local search parameters must be > 0.'

    # Check with too many elite sites
    params[param_name] = (params['n_scout_bees'] * 2, 20)

    with pytest.raises(ValueError) as e_info:
            SimpleBeesContinuous(**params)

    assert str(e_info.value) == '10 scout bees and 24 local search bees were '\
        'detected. The combination of all local bees must be less than or ' \
        'equal to `n_scout_bees`. 2 local sites were detected, please amend ' \
        'the total number of local search or scout bees.'

    invalid_input_length = [(4, 20, 2), (4,)]

    for v in invalid_input_length:
        params[param_name] = v

        with pytest.raises(ValueError) as e_info:
                SimpleBeesContinuous(**params)

        assert str(e_info.value) == f'Detected {param_name} = {v}. Please ' \
            'specify values for the number of local search sites and number '\
            f'of foraging bees. For example, {param_name} = (5, 20), ' \
            'corresponds with 5 search sites and 20 foraging sites.'
        
    # set params back to the default value
    params[param_name] = (4, 20)

#def check_site_type(param_name, params):


def test_site_params():

    params = dict(
        n_scout_bees = 10,
        bounds = (-10,10), 
        n_dim = 2,
        nbhd_radius = 1.5, 
        elite_site_params = (4, 20),
        best_site_params = (4, 20),   
    )

    # test elite params type and length. 
    # -------------------------------------------------------------------------
    check_site_value('elite_site_params', params)
    check_site_value('best_site_params', params)
    