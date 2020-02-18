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
