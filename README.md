## pybees: Python optimization toolkit using the bees algorithm

## What is it

**pybees** is a Python package for function optimization. It uses the nature inspired [bees algorithm](https://www.researchgate.net/publication/260985621_The_Bees_Algorithm_Technical_Note), proposed by Pham et al. and is built on top of SciPy. **pybees** is distributed under the 3-Clause BSD license.

The bees algorithm is a swarm based search algorithm, which mimics the food foraging behaviour of honey bees. The basic algorithm is suitable for both [continuous](https://en.wikipedia.org/wiki/Continuous_optimization) and [combinatorial](https://en.wikipedia.org/wiki/Combinatorial_optimization) optimization problems, which is demonstrated in the [basic usage](#Basic-Usage) section.


## Main features

The pybees algorithm can be used to find the global minima.




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
```

This operation returns a `scipy.optimize.optimize.OptimizeResult`. `fun` represents the value of the objective function (lowest point). `nit` represents the number of iterations taken. `x` represents the coordinates of the value found for the objective function.

```
>>> sbc.optimize(levy)
fun: 1.007063464503951e-06
nit: 100
x: array([0.99905553, 0.99967304])
```

### Combinatorial optimization (e.g. travelling salesperson problem)

```
from pybees.utils.combinatorial_single_obj import tour_distance
from pybees import SimpleBeesDiscrete

sbd = SimpleBeesDiscrete(
    n_scout_bees = 50, 
    elite_site_params = (15, 40), 
    best_site_params = (15, 30), 
    coordinates = np.random.randint(10, size=[10, 2])
)
```

This operation returns a `scipy.optimize.optimize.OptimizeResult`. `coordinates` represents a specific sequence of coordinates,  resulting from the optimization of some objective function. In this example, `tour_distance` was minimized. As such, `coordinates` represents the sequence, corresponding to the shortest path between all coordinates (i.e. travelling salesperson problem). `fun` represents the value of the objective function (shortest distance). `nit` represents the number of iterations taken. `x` represents the permutation of the original sequence passed to `SimpleBeesDiscrete` which gives the shortest distance.

```
>>> sbd.optimize(tour_distance)

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
```
