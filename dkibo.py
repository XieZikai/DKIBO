import pandas as pd

from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from bayes_opt.util import ensure_rng, acq_max
from bayes_opt.event import DEFAULT_EVENTS

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from bayes_opt.event import Events
from sklearn.ensemble import RandomForestRegressor

import warnings
from scipy.stats import norm
import copy


class DKIBO(BayesianOptimization):
    name = 'DKIBO'

    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, ml_regressor=RandomForestRegressor(n_estimators=20, max_depth=5), use_noise=False, early_stop_threshold=0.05):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()
        kernel_length_scale = np.array([1.0 for _ in range(len(pbounds))])
        if use_noise:
            kernel = Matern(nu=2.5, length_scale=kernel_length_scale) + WhiteKernel()
        else:
            kernel = Matern(nu=2.5, length_scale=kernel_length_scale)

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

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

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
            if self.ml_regressor is not None:
                self.ml_regressor.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 manual_early_stop=None,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        x_init = copy.deepcopy(self._queue._queue)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay,
                               ml_regressor=self.ml_regressor,
                               x_init=x_init,
                               max_iter=n_iter)
        iteration = 0
        x_probe_previous = None
        early_stop = False

        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params(early_stop=early_stop)
                x_probe = self.suggest(util)
                iteration += 1
                if manual_early_stop is not None and iteration >= manual_early_stop:
                    early_stop = True

            self.probe(x_probe, lazy=False)

            result = self._space.probe(x_probe)
            self.result_dataframe.append(result)

            x_probe = np.array(list(x_probe.values())) if isinstance(x_probe, dict) else x_probe
            if iteration >= init_points and self.ml_regressor is not None:
                self.proportion_dataframe.append(util.get_ucb_proportion(x_probe, self._gp))

            self.x_dataframe.append(x_probe)
            if x_probe_previous is not None and not early_stop and self.ml_regressor is not None:
                dis1 = np.linalg.norm(x_probe - np.mean(self.x_dataframe, axis=0))
                dis2 = np.linalg.norm(x_probe - x_probe_previous)
                early_stop = dis1 * self.early_stop_threshold > dis2
                if early_stop:
                    print('Early stopping activated!')

            x_probe_previous = x_probe

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)
        self.proportion_dataframe = pd.DataFrame(self.proportion_dataframe, columns=['Mean', 'Variance', 'Predictive model'])

    def get_result_dataframe(self):
        import pandas as pd
        return pd.DataFrame(np.array(self.result_dataframe))

    def get_x_dataframe(self):
        import pandas as pd
        return pd.DataFrame(np.array(self.x_dataframe))


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0, ml_regressor=None, x_init=None,
                 max_iter=100):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.ml_regressor = ml_regressor

        self.xi = xi
        self.max_iter = max_iter

        self._iters_counter = 0
        self._regression_decay_rate = 1
        self._regression_scale_rate = 1
        self._regression_scale_flag = False if ml_regressor is not None else True
        self.x_init = x_init

        self.early_stop = False

        if kind not in ['ucb', 'ei', 'poi', 'ucb_without_mean', 'regression_only']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self, early_stop=False):
        self._iters_counter += 1

        if early_stop:
            self.early_stop = True

        if self.early_stop:
            self._regression_decay_rate = 0
        else:
            self._regression_decay_rate = min(1.0, self._iters_counter ** 2 * 4. / self.max_iter ** 2)

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def get_ucb_proportion(self, x, gp):
        mean, std = gp.predict(x.reshape(1, -1), return_std=True)
        return mean, std, self.ml_regressor.predict(x.reshape(1, -1)) * self._regression_decay_rate * self._regression_scale_rate

    def utility(self, x, gp, y_max):
        result = 0
        if self.kind == 'ucb':
            result = self._ucb(x, gp, self.kappa)
            if self.ml_regressor is None:
                return result

        if self.kind == 'ei':
            result = self._ei(x, gp, y_max, self.xi)
            if self.ml_regressor is None:
                return result
            if not self._regression_scale_flag and self.x_init is not None:
                a = 0
                b = 0
                for i in self.x_init:
                    i = i.reshape(1, -1)
                    a += self._ei(i, gp, y_max, self.xi)[0]
                    b += self.ml_regressor.predict(i)[0]

                self._regression_scale_flag = True
                self._regression_scale_rate = a/b

        if self.kind == 'poi':
            result = self._poi(x, gp, y_max, self.xi)
            if self.ml_regressor is None:
                return result
            if not self._regression_scale_flag and self.x_init is not None:
                a = 0
                b = 0
                for i in self.x_init:
                    i = i.reshape(1, -1)
                    a += self._poi(i, gp, y_max, self.xi)[0]
                    b += self.ml_regressor.predict(i)[0]

                self._regression_scale_flag = True
                self._regression_scale_rate = a / b

        if self.kind == 'regression_only':
            return self.ml_regressor.predict(x)

        if self.kind == 'ucb_without_mean':
            result = self._ucb_without_mean(x, gp, self.kappa)

        return result + self.ml_regressor.predict(x) * self._regression_decay_rate * self._regression_scale_rate

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            # print('mean, std: ', mean, std)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)

    @staticmethod
    def _ucb_without_mean(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            # print('mean, std: ', mean, std)

        return kappa * std
