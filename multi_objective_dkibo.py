from dkibo import DKIBO, UtilityFunction
from deap import base
from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from bayes_opt.util import ensure_rng, acq_max
from bayes_opt.event import DEFAULT_EVENTS
from utils import NSGAII

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from bayes_opt.event import Events
from sklearn.ensemble import RandomForestRegressor

import warnings
from scipy.stats import norm
import copy


class MultiObjectiveTargetSpace(TargetSpace):

    def __init__(self, obj_num, pbounds, random_state=None):
        self.random_state = ensure_rng(random_state)

        self.obj_num = obj_num
        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0, self.obj_num))

        # keep track of unique points we have seen so far
        self._cache = {}

    def max(self):
        return self._target.max(axis=0)


class MultiObjectiveDKIBO(DKIBO):
    name = 'MODKIBO'

    def __init__(self,
                 objective_num: int,
                 pbounds,
                 random_state=None,
                 verbose=2,
                 bounds_transformer=None,
                 ml_regressor: list = None,
                 use_noise=False,
                 early_stop_threshold=0.05):

        assert objective_num == len(ml_regressor)
        self._random_state = ensure_rng(random_state)
        self._space = MultiObjectiveTargetSpace(None, pbounds, random_state)
        self.obj_num = objective_num

        self._queue = Queue()
        kernel_length_scale = np.array([1.0 for _ in range(len(pbounds[0]))])
        if use_noise:
            kernel = Matern(nu=2.5, length_scale=kernel_length_scale) + WhiteKernel()
        else:
            kernel = Matern(nu=2.5, length_scale=kernel_length_scale)

        self._gp_list = [GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        ) for _ in range(objective_num)]

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        self.ml_regressor = ml_regressor

        # recording
        self.result_dataframe = []
        self.x_dataframe = []
        self.early_stop_threshold = early_stop_threshold
        self.proportion_dataframe = []

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    def suggest(self, kind, constraints=None, n_iter=15):
        """Most promissing point to probe next"""

        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(self.obj_num):
                self._gp_list[i].fit(self._space.params, self._space.target[:, i])
                if self.ml_regressor[i] is not None:
                    self.ml_regressor[i].fit(self._space.params, self._space.target)

        suggestion = self.acq_max(
            kind=kind,
            constraints=constraints,
            n_iter=n_iter
        )

        return self._space.array_to_params(suggestion)

    def multiple_acquisition(self, kind):
        F = [None] * len(self.ml_regressor)
        for i in range(self.obj_num):



    def acq_max(self, kind,
                kappa=2.576,
                kappa_decay=1,
                kappa_decay_delay=0,
                xi=0.0,
                manual_early_stop=None,
                n_pts=100,
                n_warmup=10000, n_iter=10, constraints=None):

        bounds = self._space.bounds
        y_max = self._space.target.max()

        # acquisition functions
        utilities = []
        for i in range(self.obj_num):
            utilities.append(
                UtilityFunction(
                    kind=kind,
                    kappa=kappa,
                    kappa_decay=kappa_decay,
                    kappa_decay_delay=kappa_decay_delay,
                    xi=xi,
                    x_init=self.x_init,  # todo: adding self.x_init
                    max_iter=self.max_iter,  # todo: adding self.max_iter
                    ml_regressor=self.ml_regressor[i]
                )
            )

        def acquisitions(x):
            F = [None] * self.obj_num
            for i in range(self.obj_num):
                F[i] = utilities[i].utility(x, self._gp_list[i], y_max[i])
            return F

        acquisitions = acquisitions(utilities)

        pop, logbook, front = NSGAII(self.obj_num,
                                     acquisitions,
                                     self.pbounds,
                                     MU=n_pts)

        return suggestions