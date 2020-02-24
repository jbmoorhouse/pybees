# pybees: Python optimisation toolkit using the bees algorithm

## What is it

pybees is a Python package for function optimization. It uses the nature inspired bees algorithm, proposed by Pham et. al. and is built on top of SciPy. pybees distributed under the 3-Clause BSD license.



## Main features

## Where to get it

## Dependencies

## License

[BSD 3](LICENSE)

## Documentation

## Basic Usage

### Continuous optimization
```
from pybees.utils.continuous_single_obj import levy
from pybees import SimpleBeesContinuous

sbc = SimpleBeesContinuous(
    n_scout_bees = 50, 
    elite_site_params = (15, 40), 
    best_site_params = (15, 30),
    bounds = (-10,10), 
    n_dim = 2,
    nbhd_radius = 2
)

res = sbc.optimize(levy)
```

This operation returns a `scipy.optimize.optimize.OptimizeResult`.

```
>>> res
fun: 1.007063464503951e-06
nit: 100
x: array([0.99905553, 0.99967304])
```

### Combinatorial optimization

```
from pybees.utils.combinatorial_single_obj import tour_distance
from pybees import SimpleBeesDiscrete

sbd = SimpleBeesDiscrete(
    n_scout_bees = 50, 
    elite_site_params = (15, 40), 
    best_site_params = (15, 30), 
    coordinates = np.random.randint(10, size=[10, 2])
)

res = sbd.optimize(tour_distance)
```

This operation returns a `scipy.optimize.optimize.OptimizeResult`.

```
>>> res
coordinates: array([
       [7., 8.],
       [4., 8.],
       [2., 6.],
       [0., 5.],
       [1., 0.],
       [3., 2.],
       [2., 4.],
       [3., 4.],
       [5., 4.],
       [7., 4.]])
fun: 27.228009718084742
nit: 100
x: array([1., 9., 7., 5., 0., 3., 4., 6., 2., 8.])
