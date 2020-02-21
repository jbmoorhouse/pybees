from pybees.utils.continuous_single_obj import (
    levy, 
    ackley, 
    drop_wave, 
    michalewicz,
    easom
)

from pybees.utils.combinatorial_single_obj import tour_distance
from pybees.bees_algorithm._simple_bees_algorithm import (
    SimpleBeesContinuous, 
    SimpleBeesDiscrete,
    GLOBAL_SEARCH,
    LOCAL_SEARCH
)

import pytest
import numpy as np
from scipy.optimize import rosen


# =============================================================================
#  Complete examples and parameter lists
# =============================================================================

SBC = SimpleBeesContinuous(
    n_scout_bees = 20, 
    elite_site_params = (4,20), 
    best_site_params = (4, 10),
    bounds = (-10,10), 
    n_dim = 2,
    nbhd_radius = 1.5)

SBD = SimpleBeesDiscrete(
    n_scout_bees = 50, 
    elite_site_params = (15, 40), 
    best_site_params = (15, 30), 
    coordinates = np.random.randint(10, size=[10, 2]))


SBD_PARAMS = dict(n_scout_bees = 50, 
    elite_site_params = (15, 40), 
    best_site_params = (15, 30), 
    coordinates = np.random.randint(10, size=[10, 2])
)

# =============================================================================
#  SimpleBeesContinuous and SimpleBeesDiscrete combined tests
# =============================================================================

def check_output(sbc_obj, fun, cost, coords):
    res = sbc_obj.optimize(fun, 400)

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
    check_output(sbc_levy_4, levy, 0, [1, 1, 1, 1])

    # test drop_wave, michalewicz function with 2 dimensions only
    # -------------------------------------------------------------------------

    sbc_drop_wave_2 = SimpleBeesContinuous(
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

    sbc_easom_2 = SimpleBeesContinuous(
        n_scout_bees = 50, 
        elite_site_params = (15, 30), 
        best_site_params = (15, 20),
        bounds = (-100, 100), 
        n_dim = 2,
        nbhd_radius = 10,
    )
    
    check_output(sbc_drop_wave_2, drop_wave , -1, [0, 0])
    check_output(sbc_michalewicz_2, michalewicz, -1.8013, [2.2029, 1.5708])
    check_output(sbc_easom_2, easom, -1, [np.pi, np.pi])

    # test ackley function with 6 dimensions
    # -------------------------------------------------------------------------
    
    sbc_ackley_7 = SimpleBeesContinuous(
        n_scout_bees = 100, 
        elite_site_params = (40,100), 
        best_site_params = (30,60),
        bounds = (-32.768, 32.768),
        n_dim = 6,
        nbhd_radius = 7
    )

    check_output(sbc_ackley_7, ackley, 0, [0, 0, 0, 0, 0, 0])

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

def check_site_type(param_name, params):
    invalid_parameter_types = ["", 3, 1., 0]
    invalid_element_type = [("", 20), (4, ""), (4, 1.1), (1.1, 20)]

    # Check the parameter types
    for t in invalid_parameter_types:
        params[param_name] = t

        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(**params)

        assert str(e_info.value) == f'{param_name} must have a value of ' \
            f'type list or tuple. {type(t)} detected.'

    # Check the parameter element types
    for t in invalid_element_type:
        params[param_name] = t

        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(**params)

    # set params back to the default value
    params[param_name] = (4, 20)

def test_site_params():
    params = dict(
        n_scout_bees = 10,
        bounds = (-10,10), 
        n_dim = 2,
        nbhd_radius = 1.5, 
        elite_site_params = (4, 20),
        best_site_params = (4, 20),   
    )

    # test site parameter value type and length. 
    # -------------------------------------------------------------------------
    check_site_value('elite_site_params', params)
    check_site_value('best_site_params', params)

    # test site parameter element types 
    # -------------------------------------------------------------------------
    check_site_type('elite_site_params', params)
    check_site_type('best_site_params', params)

def test_optimize():
    # continuous no input or output
    # -------------------------------------------------------------------------

    def f():
        pass
    
    with pytest.raises(AttributeError) as e_info:
        SBC.optimize(f)
    
    assert str(e_info.value) == '`func` should accept an np.ndarray with ' \
        'shape  (dimension, n) where dimension >= 1 and n >= 1. `dimension` ' \
        'is the number of dimensions a coordinate has and n is the number of ' \
        'point  coordinates. `func` should return an np.ndarray with shape ' \
        '(m,).See the examples for SimpleBeesContinuous()'

    # continuous no output
    # -------------------------------------------------------------------------

    def g(x):
        pass
    
    with pytest.raises(TypeError) as e_info:
        SBC.optimize(g)
    
    assert str(e_info.value) == '`func` return must be an np.ndarray. ' \
        'Detected None'

    # continuous incorrect output shape
    # -------------------------------------------------------------------------

    def h(x):
        return x
    
    with pytest.raises(ValueError) as e_info:
        SBC.optimize(h)
    
    assert str(e_info.value) == 'Bad output shape (20, 2). `func` should return an array with shape (n, ) where n is the number of point coordinates. Please see the example functions. E.g. func(np.random.randint(10, size = [10, 5])) should return shape (10,).'


    # discrete incorrect number of inputs 
    # -------------------------------------------------------------------------

    invalid_func_inputs = [lambda x: None, lambda x, y, z: None]
    
    for fu in invalid_func_inputs:
        with pytest.raises(AttributeError) as e_info:
            SBD.optimize(fu)
        
        assert str(e_info.value) == '``func`` should accept 2 parameters. ' \
            '``bee_permutations`` should be an np.ndarray with shape ' \
            '``(n_permutations, n_coordinates)``, which represents ' \
            '``n_permutations`` of some ``range(coordinates)``. For example '\
            '``np.array([0,1,2], [2,1,0])`` where ``n_permutations = 2``. ' \
            'The second parameter, ``coordinates``, is an np.ndarray with ' \
            'shape ``(n_coordinates, n_dimensions)``. ``func`` should return '\
            'an np.ndarray with shape ``(n_permutations,)``. Please see ' \
            '``combinatorial_single_obj.py`` for examples.'

    # discrete incorrect output type
    # -------------------------------------------------------------------------

    with pytest.raises(TypeError) as e_info:
        SBD.optimize(lambda x, y: "")
    
    assert str(e_info.value) == '``cost_function`` should return an np.ndarray'

    with pytest.raises(TypeError) as e_info:
        SBD.optimize(lambda x, y: np.array([""]))

    assert str(e_info.value) =='``cost_function`` should return an ' \
        'np.ndarray with elements of type int or float'

    # discrete incorrect output shape
    # -------------------------------------------------------------------------

    with pytest.raises(ValueError) as e_info:
        SBD.optimize(lambda x, y: np.arange(SBD.n_scout_bees * 2))
    
    assert str(e_info.value) == f'Bad shape ({SBD.n_scout_bees * 2},). func ' \
        'should return np.ndarray with shape (n_permutations,).'

    # Check correct shape but incorrect dimensions.
    # int(sbd.n_scout_bees/2) is used so that the size is correct, but the 
    # number of dimensions is incorrect.
    with pytest.raises(ValueError) as e_info:
        SBD.optimize(lambda x, y: np.random.randint(
            10, size=[2, int(SBD.n_scout_bees/2)]))
    
    # continuous incorrect output shape
    # -------------------------------------------------------------------------
    with pytest.raises(ValueError) as e_info:
        SBC.optimize(rosen)

    assert str(e_info.value) == 'Bad output shape (2,). `func` should ' \
        'return an array with shape (n, ) where n is the number of point ' \
        'coordinates. Please see the example functions. E.g. ' \
        'func(np.random.randint(10, size = [10, 5])) should return shape (10,).'

    with pytest.raises(ValueError) as e_info:
        SBC.optimize(sum)

    assert str(e_info.value) == 'Bad output shape (2,). `func` should ' \
        'return an array with shape (n, ) where n is the number of point ' \
        'coordinates. Please see the example functions. E.g. ' \
        'func(np.random.randint(10, size = [10, 5])) should return shape (10,).'

    # continuous incorrect output type
    # -------------------------------------------------------------------------
    np.random.seed(0)

    with pytest.raises(TypeError) as e_info:
        SBC.optimize(list)

    # Check optimize n_iterations
    # -------------------------------------------------------------------------
    with pytest.raises(TypeError) as e_info:
        SBC.optimize(levy, "")
    
    assert str(e_info.value) == '``n_iter`` must be of type ``int``'

    with pytest.raises(ValueError) as e_info:
        SBC.optimize(levy, 0)

    assert str(e_info.value) == '``n_iter`` must be greater than 0'

def test_plot():
    # Test that optimize has been executed
    # -------------------------------------------------------------------------

    with pytest.raises(AttributeError) as e_info:
        SBC.plot()

    assert str(e_info.value) =='No data detected. Please execute self.optimize'

    with pytest.raises(AttributeError) as e_info:
        SBD.plot()

    assert str(e_info.value) =='No data detected. Please execute self.optimize'

    # Continuous check global_min in plot
    # -------------------------------------------------------------------------

    SBC.optimize(levy)
    
    invalid_types = [[], {}, ""]

    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            SBC.plot(global_min = t)

    invalid_length = [(0,), (0, 0, 0)]

    for l in invalid_length:
        with pytest.raises(ValueError) as e_info:
            SBC.plot(global_min = l)

    invalid_element_types = [("", 0), (0, "")]
        
    for e in invalid_element_types:
        with pytest.raises(TypeError) as e_info:
            SBC.plot(global_min = e)

    # Continuous check pad in plot
    # -------------------------------------------------------------------------

    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            SBC.plot(pad = t)

    with pytest.raises(ValueError) as e_info:
        SBC.plot(pad = -0.1)

# =============================================================================
#  SimpleBeesContinuous exclusive tests
# =============================================================================

def test_bounds():
    params = dict(
        n_scout_bees = 10,
        n_dim = 2,
        nbhd_radius = 1.5, 
        elite_site_params = (4, 20),
        best_site_params = (4, 20),   
    )

    # Check type of bounds
    invalid_type = ["", 1, 1.0, []]

    for t in invalid_type:
        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(bounds = t, **params)

            assert str(e_info.value) == f'bounds must be of type tuple. ' \
                'Detected {t}'

    # Check the length of bounds 
    invalid_length = [(-10,), (-10, 10, -10)]

    for l in invalid_length:
        with pytest.raises(ValueError) as e_info:
            SimpleBeesContinuous(bounds = l, **params)
            
        assert str(e_info.value) == 'bounds must have two values only. ' \
            f'Detected {len(l)}.'
    
    # Check the type of the bounds elements
    for b in [("", 10), (-10, ""), ("", "")]:
        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(bounds = b, **params)

    # Test that varmin is less than varmax
    with pytest.raises(ValueError) as e_info:
        SimpleBeesContinuous(bounds = (10, -10), **params)
    
    assert str(e_info.value) == 'varmax must be greater than varmin. ' \
        'Received (10, -10). Consider (-10, 10)'

    # Check the warning when varmin is equal to varmax
    with pytest.warns(UserWarning) as w_info:
        SimpleBeesContinuous(bounds=(10, 10), **params)

    assert str(w_info[0].message.args[0]) == 'Detected bounds = (10, 10). ' \
        'Consider changing bounds so that the upper bound is greater than ' \
        'the lower bound'

def test_n_dim():
    params = dict(
        n_scout_bees = 10,
        bounds = (-10, 10),
        nbhd_radius = 1.5, 
        elite_site_params = (4, 20),
        best_site_params = (4, 20),   
    )

    # Check types
    invalid_types = ["", 1.0, [], ()]

    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(n_dim = t, **params)

        assert str(e_info.value) == 'n_dim must be of type int'

    # Check values
    with pytest.raises(ValueError) as e_info:
        SimpleBeesContinuous(n_dim = 0, **params)

        assert str(e_info.value) == 'n_dim must be greater than 1'

def test_nbhd_radius():
    params = dict(
        n_scout_bees = 10,
        elite_site_params = (4, 20),
        best_site_params = (4, 20),
        bounds = (-10, 10),
        n_dim = 2  
    )

    invalid_types = ["", [], ()]

    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(**params, nbhd_radius = t)

        assert str(e_info.value) == 'nbhd_radius must be of type int or float'

    with pytest.raises(ValueError) as e_info:
        SimpleBeesContinuous(**params, nbhd_radius = 0)

    assert str(e_info.value) == 'nbhd_radius must be greater than 0'

def test_nbhd_decay():
    params = dict(
        n_scout_bees = 10,
        elite_site_params = (4, 20),
        best_site_params = (4, 20),
        bounds = (-10, 10),
        n_dim = 2,
        nbhd_radius = 1.5   
    )

    # Check type
    invalid_types = [1, ""]

    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(nbhd_decay = t, **params)

        assert str(e_info.value) == 'nbhd_decay must be of type float'
        
    invalid_values = [-0.1, 0., 1.1]

    for v in invalid_values:
        with pytest.raises(ValueError) as e_info:
            SimpleBeesContinuous(nbhd_decay = v, **params)

        assert str(e_info.value) == 'nbhd_decay must satisfy ' \
            '0 < nbhd_decay <= 1'

def test_strict_bounds():
    params = dict(
        n_scout_bees = 10,
        elite_site_params = (4, 20),
        best_site_params = (4, 20),
        bounds = (-10, 10),
        n_dim = 2,
        nbhd_radius = 1.5   
    )

    # Check type
    invalid_types = [1, "", 1.1]

    for t in invalid_types:
        with pytest.raises(TypeError) as e_info:
            SimpleBeesContinuous(strict_bounds = t, **params)

        assert str(e_info.value) == 'strict_bounds must be either True or ' \
            f'False. Received {t}'

# =============================================================================
#  SimpleBeesDiscrete exclusive tests
# =============================================================================

def test_global_search():
    global_search_keys = [k for k in GLOBAL_SEARCH]

    for key in global_search_keys:
        try:
            SimpleBeesDiscrete(**SBD_PARAMS, global_search = key)
        except ValueError:
            pytest.fail('Invalid global search method')

    invalid_types = [1.0, "", []]

    for t in invalid_types:
        with pytest.raises(TypeError):
            SimpleBeesDiscrete(**SBD_PARAMS, global_search= t)

def test_local_search():
    local_search_keys = [k for k in LOCAL_SEARCH]

    for key in local_search_keys:
        try:
            SimpleBeesDiscrete(**SBD_PARAMS, local_search = key)
        except ValueError:
            pytest.fail('Invalid local search method')

    invalid_types = [1.0, "", []]

    for t in invalid_types:
        with pytest.raises(TypeError):
            SimpleBeesDiscrete(**SBD_PARAMS, local_search = t)