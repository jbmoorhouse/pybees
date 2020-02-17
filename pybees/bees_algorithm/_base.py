"""
This module gathers the base class for the bees algorothm, 
proposed by Pham, D.T. et al.
"""

# Authors: Joseph Moorhouse <moorhouse@live.co.uk>
#
# License: BSD 3 clause

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

# =============================================================================
# Base bees algorithm
# =============================================================================


class BaseBeesAlgorithm(BaseEstimator, metaclass=ABCMeta):
    """Base class for bees algorithm.

    Warning: This class should not be used directly.
    Use derived classes instead.

    References
    ----------
    [1] Pham, D.T., S. Otri, E. Koc, A. Ghanbarzadeh, S. Rahim, and M. Zaidi. 
    2005. The Bees Algorithm. Cardiff, UK: Manufacturing Engineering Centre, 
    Cardiff University.
    """

    # -------------------------------------------------------------------------
    # Constructors

    @abstractmethod
    def __init__(self,
                 n_scout_bees,
                 **local_params):

        if not isinstance(n_scout_bees, int):
            raise TypeError('`n_scout_bees` must be of type `int`')
        elif n_scout_bees < 2:
            raise ValueError(f"Detected {n_scout_bees} scout bees. "
                             "``n_scout_bees`` must be > 2")

        self.n_scout_bees = n_scout_bees
        self.local_params = local_params

        # Check the local site parameters, including the data passed and the
        # data type used to store the parameters.
        for k, v in local_params.items():
            if not isinstance(v, (list, tuple)):
                raise TypeError(f"{k} must have a value of type list or tuple. "
                                f"{type(v)} detected.")
            elif len(v) != 2:
                raise ValueError(f"Detected {k} = {v}. Please specify values "
                                 "for the number of local search sites and "
                                 f"number of foraging bees. For example, {k} = "
                                 f"({int(n_scout_bees/2)}, 20), corresponds "
                                 f"with {int(n_scout_bees/2)} search sites and "
                                 "20 foraging sites.")

            for param in v:
                if not isinstance(param, int):
                    raise TypeError("local search parameters must be of type "
                                    f"int. type {type(param)} detected in {k}.")
                elif param < 1:
                    raise ValueError(f"Detected {param} in {k}. All local "
                                     "search parameters must be > 0.")

            setattr(self, k, v)

        self.param_grid = self._param_grid(local_params)
        self.n_local_sites = self.param_grid[0, -1]

        # Check that the number of local search bees is less than or equal to
        # the total number of scout bees. The local searhc bees must not exceed
        # the n_scout_bees.
        if n_scout_bees < self.n_local_sites:
            raise ValueError(f"{n_scout_bees} scout bees and "
                             f"{self.n_local_sites} local search bees were "
                             "detected. The combination of all local bees "
                             "must be less than or equal to `n_scout_bees`. "
                             f"{len(local_params)} local sites were detected, " 
                             "please amend the total number of local search "
                             "or scout bees.")

        # Number of bees dedicated to global search after the initial search.
        self.global_bees = n_scout_bees - self.param_grid[0, -1]

    # -------------------------------------------------------------------------
    # utility functions

    def _evaluate_bees(self, func, bees, *args):
        """Evaluate the function .

        Utility function used to evaluate the function, `func`, at the 
        coordinates in the `bees` np.ndarray. The bees coordinates and 
        fitness value are combined in the return.
        """

        bees_eval = func(bees, *args) if args else func(bees)

        if bees_eval.ndim == 1:
            bees_eval = bees_eval[:, np.newaxis]

        return np.hstack((bees, bees_eval))

    def _initial_scout_search(self, func):
        """Perform the initial scout bee search.

        For problems which consider a continuous hypercube search space, the 
        initial search consists of generating ``n_scout_bees`` with random 
        coordinates in this space. Each scout bee is subsequently evaluated for 
        fitness and ranked in ascending order, according to the fitness 
        evaluations. For problems which consider discrete coordinates, the 
        initial search consists of generating ``n_scout_bees`` different 
        permutations of the coordinates supplied. These again are sorted based 
        on their fitness with some cost function, such as the total distance 
        between each coordinate.From this search, the elite, best and new global 
        search sites are promoted to the next step in the optimisation process.
        """

        scout_bee_search = self._global_search(func, self.n_scout_bees)

        return self._sort_bees(scout_bee_search)

    def _param_grid(self, params):
        """Generate local sites parameter grid"""

        grid = np.array(list(params.values()))
        grid[:, 0] = grid[:, 0].cumsum()

        return grid.T

    def _sort_bees(self, bees):
        """sort the bees based on fitness in ascending order"""

        return bees[bees[:, -1].argsort()]

    # -------------------------------------------------------------------------
    # User defined methods
    # These methods must be defined when inheriting from BaseBeesAlgorithm

    @abstractmethod
    def _local_search(self):
        """Search the local sites for "better" solutions [1].

        Depending on the implementation, the local sites will consist of n local
        search sites. For example, for the simple bee's algorithm, there are
        elite and best local search sites. The algorithm will perform different
        tasks at these sites. Each site will have a group of foraging bees,
        which search for better solutions.

        This method should have no return. Instead it should define the local 
        search sites and subsequently search each site using the parameters 
        defined by the user. If a better solution is found in the local search 
        site, this should replace the current site.

         Please see the SimpleBeesContinuous or SimpleBeesDiscrete for examples.
        """
        pass

    @abstractmethod
    def _global_search(self):
        """Randomized search in global search space [1]

        Global search serves two roles in the bee's algorithm. It defines the 
        initial scout bee search and is used for each optimisation step for 
        the global search bees.

        This method should return a 2D np.ndarray, which contains the 
        coordinates for continuous problems and permuations for the discrete
        problems. The array should also contain the fitness cost, which should
        be concatenated row-wise. 
        
        Please see the SimpleBeesContinuous or SimpleBeesDiscrete for examples.
        """
        pass

    @abstractmethod
    def _waggle_dance(self):
        """Perform the waggle dance [1]

        This stage simulates the waggle dance performed by honey bees. This 
        methods should not have a return value, but should instead replace a 
        bee in the inital scout bee search, if the a better solution is found.

        Please see the SimpleBeesContinuous or SimpleBeesDiscrete for examples.
        """
        pass

    @abstractmethod
    def optimize(self):
        """Perform the optimization stage [1]"""
        pass
