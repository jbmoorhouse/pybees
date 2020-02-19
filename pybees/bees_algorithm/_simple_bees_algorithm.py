"""
This module gathers optmisation methods, related to the bees algorithm 
proposed by Pham, D.T. et al.
"""

# Authors: Joseph Moorhouse <moorhouse@live.co.uk>
#
# License: BSD 3 clause

import warnings

import numpy as np
import scipy
import pandas as pd 

from pybees.bees_algorithm._base import BaseBeesAlgorithm
import pybees.utils.combinatorial_search as cs
from pybees.utils.combinatorial_single_obj import tour_distance
from pybees.utils.validation import (
    check_coordinate_array,
    check_discrete_func,
    check_continuous_func,
    check_iterations
)

from tqdm import trange

import plotly.graph_objects as go
import plotly.express as px

__all__ = [
    "SimpleBeesContinuous",
    "SimpleBeesDiscrete"
]

# =============================================================================
# Types and Constants
# =============================================================================

LOCAL_SEARCH = {
    0: cs.swap,
    1: cs.reversion,
    2: cs.insertion
}

GLOBAL_SEARCH = {
    0: 'permute',
    1: 'nearest_neighbor'
}

# =============================================================================
# Simple Bees algorithm for continuous problems
# =============================================================================


class SimpleBeesContinuous(BaseBeesAlgorithm):
    """Find the global minimum of a function for continuous function.

    The bees algorithm is a two-phase method that combines a global search
    algorithm and a local minimization search at each step. The algorithm is 
    inspired by the natural foraging behaviour of honey bees [1]. 

    The algorithm begins by initializing a population of ``n_scout_bees`` in a 
    hypercube, with N-dimensions. The hypercube is searched randomly, forming
    the first global-search. The scout-bees are sorted in ascending order, based 
    on their fitness evaluation. The m best sites are divided into 2 subgroups: 
    "e" ``ellte_sites`` and "m-e" ``best_sites``. "nep" recruited-bees randomly 
    search the elite positions and "nsp" recruited-bees randomly search the 
    non-elite positions. The neighborhood search hypercube is contrained by 
    the size of the neighborhood, ngh. 

    The "n-m" non-best sites are selected for a new global search step, where
    the search space is searched randomly again. This completes a single 
    search step and is repeated until some stopping criteria is met.

    Please see the references for a more complete description

    Parameters
    ----------
    n_scout_bees: int
        The number of scout bees to use. This paramter is used to initialise 
        a population of search bees at the beginning of the algorithm.

    elite_site_params: array-like with shape (n_params,).
        `n_params` is the total number of paramaters to be considered. For the 
        simple continuous algorithm, array contains the number of "elite sites" 
        "foraging bees" respectively. The order of the parameters is critical.

    best_site_params: array-like with shape (n_params,).
        `n_params` is the total number of paramaters to be considered. For the 
        simple continuous algorithm, array contains the number of "best sites" 
        "foraging bees" respectively. The order of the parameters is critical.

    varmin: int or float
        The lower bound for each dimension in the hypercube

    varmax: int or float
        The upper bound for each dimension in the hypercube

    n_dim: int
        The number of dimensions to consider when optimizing a given function.
        `n_dim` must be greater than 0. For example, if `n_dim` is 2, then the
        algorithm is considering 2 dimensions (i.e. x and y) and optmising 
        these variables fora 3rd dimension (i.e. z).  

    nbhd_radius: int or float
        The neighbourhood search radius. Defines the local search hyperspace 
        for the elite and best sites. If `strict_bounds` is set to True, 
        bees are only permited to search within this radius if it does not
        breach the upper/lower bounds. If `strict_bounds` is set to False, 
        then the bees can search outside the upper/lower bounds. 

    nbhd_decay: float, default = 0.95
        Shrinkage constant. Following each iteration, the search space is 
        shrunk [2]. This has been shown to improve local search convergence.

    strict_bounds: bool, default = False
        Allow the algorithm to search outside `bounds`. Setting this to False
        improves time complexity. 

    Attributes
    ----------
    param_grid: np.ndarray with shape (local_params, n_local_params)
        Local parameters as an np.ndarray. local_params distinguishes the local
        sites, such as elite-site and best-sites. n_local_params denotes the
        number of site parameters, such as the number of elite site and the 
        number of elite site foraging bees. The first row represents the indices
        seperating the elite and best sites. This row is created by applying a
        cumulative sum to the number of local sites

    global_bees: int
        The number of global seach bees. Equivalent to the total number of 
        scout bees minus the total local bees (elite + best).

    References
    ----------
    [1] Pham, D.T., S. Otri, E. Koc, A. Ghanbarzadeh, S. Rahim, and M. Zaidi. 
    2005. The Bees Algorithm. Cardiff, UK: Manufacturing Engineering Centre, 
    Cardiff University.
    [2] Honey Bees Inspired Optimization Method: The Bees Algorithm
    B Yuce, M Packianather, E Mastrocinque, D Pham, A Lambiase, 2013

    Examples
    --------
    # Simple example

    >>> from bees_optimisation import SimpleBeesContinuous
    ...
    >>> sbc = SimpleBeesContinuous(
            n_scout_bees = 15, 
            elite_site_params = (10, 10), 
            best_site_params = (5, 5),
            bounds = (-10, 10), 
            n_dim = 2,
            nbhd_radius = 3,
        )
    ...

    # Show the parameter grid.

    >>> sbc.param_grid
    ...
    array([[10, 15],
           [10,  5]])
    """

    # -------------------------------------------------------------------------
    # Constructors

    def __init__(self,
                 n_scout_bees,
                 elite_site_params,
                 best_site_params,
                 bounds,
                 n_dim,
                 nbhd_radius,
                 nbhd_decay=0.95,
                 strict_bounds=False):

        super().__init__(
            n_scout_bees=n_scout_bees,
            elite_site_params=elite_site_params,
            best_site_params=best_site_params
        )

        # Check that bounds is passed as a tuple and with only two values.
        if not isinstance(bounds, tuple):
            raise TypeError(f"bounds must be of type tuple. Detected {bounds}")
        elif len(bounds) != 2:
            raise ValueError("bounds must have two values only. Detected "
                             f"{len(bounds)}.")

        # Check each bound is either an int or float
        for v in bounds:
            if not isinstance(v, (int, float)):
                raise TypeError("bounds must only be of type int or float. "
                                f"Detected {v}")

        self.varmin, self.varmax = bounds

        # Check that varmax is greater than varmin
        if self.varmax < self.varmin:
            raise ValueError("varmax must be greater than varmin. Received "
                             f"{bounds}. Consider {bounds[::-1]}")
        elif self.varmin == self.varmax:
            warnings.warn(f"Detected bounds = {bounds}. Consider changing "
                          "bounds so that the upper bound is greater than the "
                          "lower bound", UserWarning)

        # Check that the number of dimensions is an integer value and greater
        # than 1. This is a practical check.
        if not isinstance(n_dim, int):
            raise TypeError("n_dim must be of type int")
        elif n_dim < 1:
            raise ValueError("n_dim must be greater than 1")

        self.n_dim = n_dim

        # Check that the radius is either an integer or floating point value
        # and is also greater than zero.
        if not isinstance(nbhd_radius, (int, float)):
            raise TypeError("nbhd_radius must be of type int or float")
        elif nbhd_radius <= 0:
            raise ValueError("nbhd_radius must be greater than 0")

        self.nbhd_radius = nbhd_radius

        # Check that the shrinkage rate is a floating point value and is
        # between 0 and 1, exclusive and inclusive respectively.
        if not isinstance(nbhd_decay, float):
            raise TypeError("nbhd_decay must be of type float")
        elif not 0 < nbhd_decay <= 1:
            raise ValueError("nbhd_decay must satisfy 0 < nbhd_decay <= 1")

        self.nbhd_decay = nbhd_decay

        if not isinstance(strict_bounds, bool):
            raise TypeError("strict_bounds must be either True or False. "
                            f"Received {strict_bounds}")

        self.strict_bounds = strict_bounds

    # ------------------------------------------------------------------------
    # Search methods

    def _local_search(self, func, bees, nbhd):
        # Get parameter grid and select the local sites. The entries indicate
        # where along axis the array is split. For example, arr = [2, 3]
        # would, for axis=0, result in arr[:2], arr[2:3], ary[3:]. The last
        # split group is omitted, since these are the global search bees.
        grid = self.param_grid
        local_sites = np.split(ary=bees, indices_or_sections=grid[0])[:-1]

        # For each local site, select the number of forager bees from `grid`
        # and apply perform the `_waggle_dance`.
        for idx, site in enumerate(local_sites):
            forager_bees = grid[1, idx]
            np.apply_along_axis(
                func1d=self._waggle_dance,
                axis=1,
                arr=site,
                func=func,
                n_bees=forager_bees,
                nbhd=nbhd)

    def _global_search(self, func, n_bees):
        global_search_coords = np.random.uniform(
            self.varmin, self.varmax, [n_bees, self.n_dim])

        return self._evaluate_bees(func, global_search_coords)

    def _waggle_dance(self, bee, func, n_bees, nbhd):
        forage = bee[:-1] + \
            np.random.uniform(-nbhd, nbhd, [n_bees, self.n_dim])

        # Limit the search space is strict bounds is enforced
        if self.strict_bounds:
            forage = np.clip(forage, self.varmin, self.varmax)

        forage_cost = self._evaluate_bees(func, forage)
        top_forager = forage_cost[forage_cost[:, -1].argmin()]

        np.copyto(bee, top_forager, where=top_forager[-1] < bee[-1])

    # ----------------------------------------------------------------------
    # Public methods

    def optimize(self, func, n_iter=100):
        """Find the global minimum of a function using the bees algorithm

        Parameters
        ----------
        func : callable ``f(x)``
            Function to be optimized. `func` must accept an np.ndarray with
            shape(n_coordinates, n_dim) and return a np.ndarray with shape
            (n_coordinates,)

        n_iter : int, default = 100
            Number of optimisation steps 

        Returns
        -------
        res : np.ndarray, with shape (n_coordinates,)
            The optimisation result, represented as an np.ndarray. The 
            result contains the coordinates, where the lowest minimum
            was found.

        Examples
        --------
        # Simple example


        >>> from bees_optimisation import SimpleBeesContinuous
        ...
        ...
        >>> def ackley(x, a = 20, b = 0.2, c = 2 * PI):
                x = x.reshape(-1, x.size) if x.ndim == 1 else x
                d = x.shape[1]
                    
                sum_one = np.sum(x ** 2, axis=1)
                sum_two = np.sum(np.cos(c * x), axis=1)
                
                term_one = -a * np.exp(-b * ((sum_one/d) ** 0.5))
                term_two = -np.exp(sum_two/d)

                return term_one + term_two + a + np.exp(1)
        ...
        ...
        >>> sbc = SimpleBeesContinuous(
                n_scout_bees = 50, 
                elite_site_params = (10,100), 
                best_site_params = (10,60),
                bounds = (-32.768, 32.768),
                n_dim = 1,
                nbhd_radius = 10
            )
        ...
        >>> sbc.optimize(ackley)
        array([4.31090466e-08,  6.50332713e-08, -4.84183941e-08])
        """

        # Check the continuous function accepts and returns the correct
        # data types and shapes
        check_continuous_func(func, self.n_dim)

        # Check n_iter input is correct
        n_iter = check_iterations(n_iter)

        # Begin the initial global search with scout bees
        scout_sorted = self._initial_scout_search(func)
        self.history = scout_sorted.copy()

        nbhd = self.nbhd_radius

        with trange(n_iter) as pbar:
            for _ in pbar:
                # Get the curarent best cost
                pbar.set_postfix(cost=scout_sorted[0, -1])

                self._local_search(func, scout_sorted, nbhd)

                if self.global_bees > 0:
                    scout_sorted[self.param_grid[0, -1]:] = (
                        self._global_search(func, self.global_bees))

                # Save the current state
                self.history = np.concatenate(
                    (self.history, scout_sorted.copy()))

                # Set new neighbourhood search hyper space.
                nbhd *= self.nbhd_decay
                scout_sorted = self._sort_bees(scout_sorted)

        # Prepare the results and return object
        res = scipy.optimize.OptimizeResult()

        res.x = scout_sorted[0, :-1]
        res.fun = scout_sorted[0, -1]
        res.nit = n_iter

        return res

    def plot(self, global_min=None, pad=5):
        """Track the bee position 

        The history of the bees position is tracked and plotted.

        Parameters
        ----------
        global_min : tuple, default = None
            Plot the coordinates of the global minimum
        pad : int, default = 5
            The percentage by which to pad the plot limits. I.e. if the 
            search space is in ``bounds = (-10, 10)`` then the plotting
            limits is set to ``(-10.5 , 10.5)``. 

        Notes
        -----
        The plotting function is currently only recommended when 
        ``n_dim = 2``. While the plotting functionality works when
        ``n_dim != 2``, it is not advised.

        Raises
        ------
        AttributeError
            If ``.optimize(func)`` has not been executed.
        """

        # Determine if optimize has been executed
        if not hasattr(self, "history"):
            raise AttributeError("No data detected. Please execute "
                                 "use the optimise method")

        # Check global_min is the correct shape and size
        if not isinstance(global_min, tuple):
            raise TypeError(f"Bad type {type(global_min)}")
        elif len(global_min) != 2:
            raise ValueError("Bad size detected")

        # Check that all values passed to global_min are correct.
        for m in global_min:
            if not isinstance(m, (int, float)):
                raise TypeError(f"Bad type {type(m)}. global_min must "
                                "contain integers of floating points only.")

        # Check pad type and value
        if not isinstance(pad, (int, float)):
            raise ValueError(f"Bad type {type(pad)}. Must be int or float")
        elif pad < 0:
            raise ValueError(f"Bad size {pad}, pad must be greater than 0.")

        # DataFrame column names
        xi = [f"x{i + 1}" for i in range(self.history.shape[1] - 1)]
        # Define the DataFrame
        df = pd.DataFrame(self.history, columns=[*xi, 'cost'])

        # Set animation columns
        df['n_iter'] = np.repeat(
            np.arange(int(df.shape[0]/self.n_scout_bees)), self.n_scout_bees)
        df['bee'] = np.tile(
            np.arange(self.n_scout_bees), int(df.shape[0]/self.n_scout_bees))

        pad_by = max(self.varmin, self.varmax) * (pad/100)
        r = [self.varmin - pad_by, self.varmax + pad_by]

        # Scatter plot
        fig = px.scatter(df, x=xi[0], y=xi[1], animation_frame='n_iter',
                         animation_group="bee", color_continuous_scale='viridis',
                         color='cost', height=1000, range_x=r, range_y=r)

        # Add global minimum
        if global_min:
            fig.add_trace(go.Scatter(
                x=np.array([global_min[0]]),
                y=np.array([global_min[1]]),
                marker=dict(color='Red', size=15, symbol='cross')))

        fig.show()

# =============================================================================
# Simple Bees algorithm for discrete problems
# =============================================================================


class SimpleBeesDiscrete(BaseBeesAlgorithm):
    """Find the global minimum of a function for combinatorial cases.

        The bees algorithm is a two-phase method that combines a global search
    algorithm and a local minimization search at each step. The algorithm is 
    inspired by the natural foraging behaviour of honey bees [1]. 

    The algorithm considers a sequence of discrete coordinates. The algorithm 
    begins by initializing a population of ``n_scout_bees`` in the first 
    global search stage. The global search consist of generating 
    ``n_scout_bees`` permuations of the discrete coordinates, using a defined
    global search strategy. The scout-bees are sorted in ascending order, based 
    on their fitness evaluation. The "m" best sites are divided into 2 
    subgroups: "e" ``ellte_sites`` and "m-e" ``best_sites``. "nep" 
    recruited-bees randomly search the elite sites and "nsp" recruited-bees 
    randomly search the best sites. These local search methods include mutation 
    steps such as swap, reversion or insertion [2]. Any local search 
    improvments are stored.

    The "n-m" non-best sites are selected for a new global search step. This 
    completes a single search step and is repeated until some stopping criteria 
    is met. Please see the references for a more complete description.

    Parameters
    ----------
    n_scout_bees: int
        The number of scout bees to use. This paramter is used to initialise 
        a population of search bees at the beginning of the algorithm.

    elite_site_params: array-like with shape (n_params,).
        `n_params` is the total number of paramaters to be considered. For the 
        simple discrete algorithm, the tuple contains the numer of elite sites
        and the number of elite foraging bees, in this order. For example, for
        10 elite bees and 20 elite foraging bees per site, then
        ``elite_site_params = (10, 20)``.

    best_site_params: array-like with shape (n_params,).
        `n_params` is the total number of paramaters to be considered. For the 
        simple discrete algorithm, the tuple contains the numer of best sites
        and the number of best foraging bees, in this order. For example, for
        10 best bees and 15 best foraging bees per site, then
        ``best_site_params = (10, 15)``.

    coordinates: np.ndarray with shape(n_coordinates, n_dimensions) 
        Array with contains the discrete coordinates. A single coordinate 
        is is represented row-wise. Pleas see the examples.

    global_search: int, default = None
        Set the global search method. If None, global search methods are 
        sampled randomly from GLOBAL_SEARCH.

        * If 0, then generate a random permuttion of the coordinates   
        * If 1, then use the nearest neighbour algorithm. 

        Note: Other methods will be added at a later date

    local_search: int, default = None
        Set the local search method. If None, local search methods are 
        sampled randomly from LOCAL_SEARCH.

        * If 0, then use swap mutation.   
        * If 1, then use reversion mutation.
        * If 2, then use insertion mutation. 

    References
    ----------
    [1] Pham, D.T., S. Otri, E. Koc, A. Ghanbarzadeh, S. Rahim, and M. Zaidi. 
    2005. The Bees Algorithm. Cardiff, UK: Manufacturing Engineering Centre, 
    Cardiff University.
    [2] Jolai, F., Rabiee, M. and Asefi, H. (2012). A novel hybrid 
    meta-heuristic algorithm for a no-wait flexible flow shop scheduling 
    problem with sequence dependent setup times. International Journal of 
    Production Research, 50(24), pp.7447–7466.
    """

    # -------------------------------------------------------------------------
    # Constructors

    def __init__(self,
                 n_scout_bees,
                 elite_site_params,
                 best_site_params,
                 coordinates,
                 global_search=None,
                 local_search=None):

        super().__init__(
            n_scout_bees=n_scout_bees,
            elite_site_params=elite_site_params,
            best_site_params=best_site_params
        )

        self.coordinates = check_coordinate_array(coordinates)
        self.global_search_funcs = cs.GlobalSearch(coordinates)

        # Overide global and local search methods
        if isinstance(global_search, int):
            if global_search not in GLOBAL_SEARCH:
                k, v = zip(*GLOBAL_SEARCH.items())

                raise ValueError("Invalid global search method selection. "
                                f"Please choose from {k}, which correspond "
                                f"to the search methods {v} respectively.")
        elif global_search is not None:
            raise TypeError(f"Bad type {type(global_search)}. Must "
                            "be of type int or NoneType.")

        self.global_search_method = global_search

        if isinstance(local_search, int):
            if local_search and local_search not in LOCAL_SEARCH:
                k, v = zip(*[(k, v.__name__) for k, v in LOCAL_SEARCH.items()])

                raise ValueError("Invalid local search method selection. "
                                f"Please choose from {k}, which correspond "
                                f"to the search methods {v} respectively.")
        elif local_search is not None:
            raise TypeError(f"Bad type {type(local_search)}. Must "
                            "be of type int or NoneType.")
        
        self.local_search_method = local_search

    # ----------------------------------------------------------------------
    # User defined local search methods

    def _local_search(self, func, bees):
        grid = self.param_grid
        local_sites = np.split(ary=bees, indices_or_sections=grid[0])[:-1]

        # For each local site, select the number of forager bees from `grid`
        # and apply perform the `_waggle_dance`.
        for idx, site in enumerate(local_sites):
            forager_bees = grid[1, idx]
            np.apply_along_axis(
                func1d=self._waggle_dance,
                axis=1,
                arr=site,
                func=func,
                n_bees=forager_bees)

    def _global_search(self, func, n_bees):
        if self.global_search_method is not None:
            global_attr = GLOBAL_SEARCH[self.global_search_method]
        else:
            m = np.random.randint(len(GLOBAL_SEARCH))
            global_attr = GLOBAL_SEARCH[m]

        global_func = getattr(self.global_search_funcs, global_attr)
        bees = global_func(n_bees)

        return self._evaluate_bees(func, bees, self.coordinates)

    def _waggle_dance(self, bee, func, n_bees):
        if self.local_search_method is not None:
            forage_func = LOCAL_SEARCH[self.local_search_method]
        else:
            m = np.random.randint(len(LOCAL_SEARCH))
            forage_func = LOCAL_SEARCH[m]

        forage = forage_func(bee[:-1], n_bees)
        forage_cost = self._evaluate_bees(func, forage, self.coordinates)
        top_forager = forage_cost[forage_cost[:, -1].argmin()]

        np.copyto(bee, top_forager, where=top_forager[-1] < bee[-1])

    # ----------------------------------------------------------------------
    # Public methods

    def optimize(self, func, n_iter=100):
        """Find the global minimum of a function using the bees algorithm

        Parameters
        ----------
        func : callable ``f(bee_permutation, coordinates)``
            Function to be optimized. `func` must accept two arguments.
            ``bee_permutation`` must be an np.ndarray, with 
            shape(n_coordinates,) and represent a permutation of 
            ``coordinates`` must be an np.ndarray with 
            shape(n_coordinates, n_dim). Please see the example below.

        n_iter : int, default = 100
            Number of optimisation steps 

        Returns
        -------
        res : np.ndarray, with shape (n_coordinates,)
            The optimisation result, represented as an np.ndarray. The 
            result contains the best permuation of coordinates.

        Examples
        --------
        # Simple example


        >>> from bees_optimisation import SimpleBeesContinuous
        >>> from combinatorial import tour_distance
        ...
        >>> coordinates = np.random.randint(-10,10, [10,2])
        >>> coordinates
        array([[-9, -1],
               [ 9, -6],
               [ 1,  8],
               [ 4, -5],
               [-9,  4],
               [-3,  6],
               [-6,  0],
               [-8,  6],
               [ 8, -1],
               [ 4,  2]])

        ...
        >>> sbd = SimpleBeesDiscrete(
                n_scout_bees = 50, 
                elite_site_params = (15, 40), 
                best_site_params = (15, 30), 
                coordinates = coordinates
            )

        >>> sbd.optimize(tour_distance)
        array([[-9., -1.],
               [-9.,  4.],
               [-8.,  6.],
               [-3.,  6.],
               [ 1.,  8.],
               [ 4.,  2.],
               [ 8., -1.],
               [ 9., -6.],
               [ 4., -5.],
               [-6.,  0.]])
        """

        # Check the discrete cost function accepts and returns the correct
        # data types and shapes
        check_discrete_func(func, self.n_scout_bees)
        
        # Check n_iter input is correct
        n_iter = check_iterations(n_iter)

        scout_sorted = self._initial_scout_search(func)
        self.history = scout_sorted.copy()[:1]

        with trange(n_iter) as pbar:
            for _ in pbar:
                # Get the curarent best cost
                pbar.set_postfix(cost=scout_sorted[0, -1])

                self._local_search(func, scout_sorted)

                if self.global_bees > 0:
                    scout_sorted[self.param_grid[0, -1]:] = (
                        self._global_search(func, self.global_bees))

                # Save the current state and sort
                self.history = np.concatenate(
                    (self.history, scout_sorted.copy()[:1]))
                scout_sorted = self._sort_bees(scout_sorted)

        # Ony return the bee coordinates
        res = scipy.optimize.OptimizeResult()

        res.x = scout_sorted[0, :-1]
        res.coordinates = self.coordinates[scout_sorted[0, :-1].astype(int)]
        res.fun = scout_sorted[0, -1]
        res.nit = n_iter

        return res

    def plot(self, title=None, height=1000, pad=5):
        """Plot the optimisation history.

        At each step, the best solution is stored in self.history. The 
        plotting method displays these best solutions and tracks the 
        optmisation steps

        Parameters
        ----------
        title : str, default = None
            Plot title

        height : int, default = 1000
            plot height

        pad : int, default = 5
            The percentage by which to pad the plot limits. I.e. if the 
            search space is in ``bounds = (-10, 10)`` then the plotting
            limits is set to ``(-10.5 , 10.5)``. 

        Notes
        -----
        The plotting function is currently only recommended when 
        ``n_dim = 2``. While the plotting functionality works when
        ``n_dim != 2``, it is not advised.

        Examples
        --------
        from combinatorial_search import tour_distance

        >>> from bees_optimisation import SimpleBeesContinuous
        >>> from combinatorial import tour_distance
        ...
        >>> coordinates = np.random.randint(-10,10, [10,2])
        ...
        >>> sbd = SimpleBeesDiscrete(
                n_scout_bees = 50, 
                elite_site_params = (15, 40), 
                best_site_params = (15, 30), 
                coordinates = coordinates
            )

        >>> sbd.optimize(tour_distance)
        >>> sbd.plot()
        """
        
        if not hasattr(self, "history"):
            raise AttributeError("No data detected. Please execute "
                                 "self.optimise")

        # Define abbreviated useful data
        c = self.coordinates
        n, m = c.shape

        if m > 3:
            raise ValueError(f"Bad shape {(n, m)}. For plotting, m must "
                             "be in range 1 < m < 4.")

        # Get the coordinates in the optimum permutation found. Reshape
        # all x, y, z etc. coordinates to a single row. This was done
        # concatenating the first coordinate easier.
        coord_history = c[self.history[:, :-1].astype(int)]
        reshape = coord_history.transpose((0, 2, 1)).reshape(-1, n)

        # Concatenate with the first coordinate to create a complete tour.
        # Then create a 2D array containing the complete coordiate history.
        # This shape was selected to align with the formatting requirements
        # imposed by plotly. A list comprehension used so it is suitable
        # for n dimensions.
        opt_coords = np.concatenate(
            (reshape, reshape[:, 0][:, np.newaxis]), 1)
        data = np.concatenate(
            [opt_coords[i::m].reshape(-1, 1) for i in range(m)],
            axis=1)

        # Define DataFrame columns and instantiate the DataFrame.
        xi = [f"x{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=xi)

        # Create animation and colouring columns
        df['n_iter'] = np.repeat(np.arange(int(df.shape[0]/(n + 1))), n+1)

        # Define the plotting border pad
        x_pad = c[:, 0].max() * (pad/100)
        y_pad = c[:, 1].max() * (pad/100)

        r_x = [c[:, 0].min() - x_pad, c[:, 0].max() + x_pad]
        r_y = [c[:, 1].min() - y_pad, c[:, 1].max() + y_pad]

        if m == 3:
            z_pad = y_pad = c[:, 2].max() * (pad/100)
            r_z = [c[:, 2].min() - y_pad, c[:, 2].max() + y_pad]

            fig = px.line_3d(df, x=xi[0], y=xi[1], z=xi[2],
                             animation_frame='n_iter', range_x=r_x, range_y=r_y,
                             range_z=r_z,  height=height, title=title)
        else:
            fig = px.line(df, x=xi[0], y=xi[1], animation_frame='n_iter',
            range_x=r_x, range_y=r_y, height=height, title=title)

        fig.show()