from typing import List

from dkibo import DKIBO, UtilityFunction
from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from bayes_opt.util import ensure_rng
from bayes_opt.event import DEFAULT_EVENTS
from utils import NSGAII

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

import warnings
import copy
from dppy.finite_dpps import FiniteDPP
import pandas as pd


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
                 objective_target: List[str],
                 pbounds,
                 random_state=None,
                 verbose=2,
                 init_points=5,
                 bounds_transformer=None,
                 ml_regressor: List = None,
                 use_noise=False,
                 early_stop_threshold=0.05,
                 constraint=None):
        """
        :params:
        objective_target: A list of optimization directions, for example ['max', 'min', 'max']
        constraint: Callable function of input vector. Only accept inequality constraints that the result smaller than 0.
        """
        assert len(ml_regressor) == len(objective_target)
        self._random_state = ensure_rng(random_state)
        self.obj_num = len(objective_target)
        self.constraint = constraint
        self.obj_target = objective_target

        self._queue = Queue()
        kernel_length_scale = np.array([1.0 for _ in range(len(pbounds))])
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
        ) for _ in range(self.obj_num)]

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        self.ml_regressor = ml_regressor
        self.x_init = None
        self.init_points = init_points

        # recording
        self.result_dataframe = []
        self.x_dataframe = []
        self.early_stop_threshold = early_stop_threshold
        self.proportion_dataframe = []
        self.pbounds = np.array(list(pbounds.values()))

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

        self._space = MultiObjectiveTargetSpace(self.obj_num, pbounds, random_state)

    def _prime_queue(self, init_points=None):
        init_points = self.init_points if init_points is None else init_points
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

        self.x_init = copy.deepcopy(self._queue._queue)

    def suggest(self, kind, batch_size=1, max_iter=10,
                save_df=False, load_df=False, delta=1e-4):
        """Most promissing point to probe next"""
        # 每次调用，max_iter数值应当减少，直到减少到2

        if self.x_init is None:
            self.x_init = []
            for i in range(self.init_points):
                self.x_init.append(self._space.random_sample())

        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(self.obj_num):
                self._gp_list[i].fit(self._space.params, self._space.target[:, i])
                if self.ml_regressor[i] is not None:
                    self.ml_regressor[i].fit(self._space.params, self._space.target)

        suggestions = self.acq_max(
            kind=kind,
            max_iter=max_iter,
            batch_size=batch_size
        )
        results = [self._space.array_to_params(suggestion) for suggestion in suggestions]

        if load_df:
            previous_df = pd.read_csv('previous_batch.csv')
            df = pd.DataFrame(results)
            if ((df - previous_df) ** 2).values.sum() < df.values.sum() * delta:
                suggestions = self.acq_max(
                    kind=kind,
                    max_iter=max_iter,
                    batch_size=batch_size,
                    early_stop=True
                )
                results = [self._space.array_to_params(suggestion) for suggestion in suggestions]

        if save_df:
            df = pd.DataFrame(results)
            df.to_csv('previous_batch.csv')

        return results

    def suggest_test(self, kind, batch_size=1, max_iter=10):
        """Most promissing point to probe next"""
        # 每次调用，max_iter数值应当减少，直到减少到2

        if self.x_init is None:
            self.x_init = []
            for i in range(self.init_points):
                self.x_init.append(self._space.random_sample())

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
            max_iter=max_iter,
            batch_size=batch_size
        )

        return self._space.array_to_params(suggestion)

    def acq_max(self, kind,
                kappa=2.576,
                kappa_decay=1,
                kappa_decay_delay=0,
                xi=0.0,
                n_pts=100,
                max_iter=100,
                early_stop=False,
                batch_size=1):

        # bounds = self._space.bounds
        y_max = self._space.max()

        # acquisition functions
        utilities = []
        for i in range(self.obj_num):
            util = UtilityFunction(
                    kind=kind,
                    kappa=kappa,
                    kappa_decay=kappa_decay,
                    kappa_decay_delay=kappa_decay_delay,
                    xi=xi,
                    x_init=self.x_init,
                    max_iter=max_iter,
                    ml_regressor=self.ml_regressor[i]
                )
            if early_stop:
                util.update_params(early_stop=True)
            utilities.append(util)

        def acquisitions(x):
            x_array = np.array(x.tolist())
            x_array = x_array.reshape(1, -1)
            result = [utilities[i].utility(x_array, self._gp_list[i], y_max[i])[0] for i in range(self.obj_num)]
            return result

        pop, logbook, front = NSGAII(acquisitions,
                                     self.pbounds,
                                     MU=n_pts,
                                     constraint=self.constraint,
                                     target=self.obj_target)

        pop = np.asarray(pop)  # shape: (n_pts, len(pbounds))
        # adding DPPs here
        kernel_matrix = [gp.kernel_(pop) for gp in self._gp_list]
        kernel_matrix = np.mean(kernel_matrix, axis=0)

        # comment DPP isin_01 before using it
        dpp = FiniteDPP('correlation', **{'K': kernel_matrix})
        suggestions = dpp.sample_exact_k_dpp(batch_size)

        return pop[suggestions]
