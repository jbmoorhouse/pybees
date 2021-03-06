## pybees: Python optimization toolkit using the bees algorithm

<br/>
<p align="center">
   <img src="https://media.giphy.com/media/yIXVnzpoNiE0w/source.gif" width="600" />
</p>
<br/>

## What is it?

**pybees** is a Python package for function optimization. It uses the nature inspired [**bees algorithm**](https://www.researchgate.net/publication/260985621_The_Bees_Algorithm_Technical_Note), proposed by Pham et al. and is built on top of SciPy. **pybees** is distributed under the 3-Clause BSD license.

The bees algorithm is a swarm based search algorithm, which mimics the food foraging behaviour of honey bees. The basic algorithm is suitable for both [continuous](https://en.wikipedia.org/wiki/Continuous_optimization) and [combinatorial](https://en.wikipedia.org/wiki/Combinatorial_optimization) optimization problems, which is demonstrated in the [**basic examples**](#Basic-Examples) section.


## Main features

- High-level API to optimize continuous functions (training a [**multilayer perceptron**](https://en.wikipedia.org/wiki/Multilayer_perceptron)).
- High-level API to optimize discrete functions ([**traveling salesperson problem**](https://en.wikipedia.org/wiki/Travelling_salesman_problem)).
- Built in single-objective cost functions.

- [**plotly**](https://plot.ly/python/) plotting tools for 2D and 3D (both continuous and discrete).
- Extensible API for developing new ideas.
    

## Where to get it

If all dependencies are already installed, then the easiest way to install pybees is using `pip`.

```python
pip install pybees
```

## How to use it

Please visit the examples folder to view the demonstration [examples](examples). These include.

- [Continuous function](https://github.com/jbmoorhouse/pybees/blob/master/examples/continuous.ipynb) optimization ([*levy*](https://www.sfu.ca/~ssurjano/levy.html), [*drop_wave*](https://www.sfu.ca/~ssurjano/drop.html) and [*michalewicz*](https://www.sfu.ca/~ssurjano/michal.html))
- [Combinatorial function](https://github.com/jbmoorhouse/pybees/blob/master/examples/combinatorial.ipynb) optimization (travelling salesperson problem)

Future examples will include

- Multilayer perceptron

## Dependencies

pybees requires:

- numpy >= 1.17.4 <br/>
- scipy >= 1.3.2 <br/>
- plotly >= 4.4.1<br/>
- tqdm >= 4.40.2<br/>
- sklearn >= 0.22<br/>
- pandas >= 0.25.3<br/>

## License

[BSD 3](LICENSE)


## Basic Examples

### Continuous function optimization 
```python
import pybees as pb

sbc = pb.SimpleBeesContinuous(
    n_scout_bees = 50, 
    elite_site_params = (15, 40), 
    best_site_params = (15, 30),
    bounds = (-10,10), 
    n_dim = 2,
    nbhd_radius = 2
)
```

This operation returns a `scipy.optimize.OptimizeResult`. `fun` represents the value of the objective function (lowest point). `nit` represents the number of iterations taken. `x` represents the coordinates of the value found for the objective function.

```python
>>> sbc.optimize(pb.levy)
fun: 1.007063464503951e-06
nit: 100
x: array([0.99905553, 0.99967304])
```

The results may also be visualised by using the following

```python
sbc.plot(global_min = (1, 1))
```

### Combinatorial optimization (e.g. travelling salesperson problem)

```python
import pybees as pb

sbd = pb.SimpleBeesDiscrete(
    n_scout_bees = 50, 
    elite_site_params = (15, 40), 
    best_site_params = (15, 30), 
    coordinates = np.random.randint(10, size=[10, 2])
)
```

This operation returns a `scipy.optimize.OptimizeResult` result. `res.coordinates` represents `sbd.coordinates` reordered, resulting from the optimization of some objective function. In this example, `tour_distance` was minimized. As such, `coordinates` represents the sequence, corresponding to the shortest path between all coordinates (i.e. travelling salesperson problem). `fun` represents the value of the objective function (shortest distance). `nit` represents the number of iterations taken. `x` represents the permutation of the original sequence passed to `SimpleBeesDiscrete` which gives the shortest distance.

```python
>>> sbd.optimize(pb.tour_distance)
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


The results may also be visualised by using the following

```python
sbd.plot()
```