from mobopt import MOBayesianOpt
from mobopt._bayes import ConstraintError
from mobopt._target_space import TargetSpace
from mobopt._wrapper import GaussianProcessWrapper as GPR
from mobopt._helpers import plot_1dgp
from mobopt._NSGA2 import NSGAII

import numpy as np
import warnings
from scipy.stats import norm
from utils import round_result

from warnings import warn
import matplotlib.pyplot as pl
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist


def sample_maximal_distance_points(points, n, initial_point_index, redundant_sampling=10):
    # using redundant sampling to avoid duplicate probe point error

    # 计算距离矩阵
    distance_matrix = cdist(points, points, metric='euclidean')

    # 选择初始点
    selected_points = [initial_point_index]

    # 迭代选择下一个点
    while len(selected_points) < n + redundant_sampling:
        # 计算当前已选择点与剩余点之间的最小距离
        min_distances = np.min(distance_matrix[selected_points], axis=0)

        # 选择与当前已选择点距离最大的点
        next_point_index = np.argmax(min_distances)

        # 添加新选择的点
        selected_points.append(next_point_index)

    # 返回选定的点
    selected_points = np.array(selected_points)
    selected_points_coordinates = points[selected_points]

    return selected_points_coordinates


class NoTargetSpace(TargetSpace):

    def __init__(self, NObj, pbounds, constraints,
                 RandomSeed, init_points=2, verbose=False):
        super().__init__(None, NObj, pbounds, constraints,
                         RandomSeed, init_points, verbose)

    def observe_point(self, x, f):
        assert x.size == self.NParam, 'x must have the same dimension'

        try:
            NewF = []
            for ff in f:
                NewF.append(ff[0])
            f = NewF
        except:  # noqa
            pass

        self.add_observation(x, f)

        return f


class NoTargetMOBayesianOpt(MOBayesianOpt):
    def __init__(self, target, NObj, pbounds, constraints=[], verbose=False, Picture=False, TPF=None,
                 n_restarts_optimizer=10, Filename=None, MetricsPS=True, max_or_min='max', RandomSeed=None,
                 kernel=None, ml_regressor=None):

        super(MOBayesianOpt, self).__init__()
        self.verbose = verbose
        self.vprint = print if verbose else lambda *a, **k: None

        self.counter = 0
        self.constraints = constraints
        self.n_rest_opt = n_restarts_optimizer
        self.Filename = Filename
        self.MetricsPS = MetricsPS

        # reset calling variables
        self.__reset__()

        # number of objective functions
        if isinstance(NObj, int):
            self.NObj = NObj
        else:
            raise TypeError("NObj should be int")

        if Picture and self.NObj == 2:
            self.Picture = Picture
        else:
            if Picture:
                warn("NObj must be 2 to plot PF convergence")
            self.Picture = False

        self.target = None

        self.pbounds = pbounds
        # pbounds must hold the bounds for each parameter
        try:
            self.NParam = len(pbounds)
        except TypeError:
            raise TypeError("pbounds is neither a np.array nor a list")
        if self.pbounds.shape != (self.NParam, 2):
            raise IndexError("pbounds must have 2nd dimension equal to 2")

        self.vprint(f"Dim. of Search Space = {self.NParam}")

        if TPF is None:
            self.vprint("no metrics are going to be saved")
            self.Metrics = False
        else:
            self.vprint("metrics are going to be saved")
            self.Metrics = True
            self.TPF = TPF

        if self.Filename is not None:
            self.__save_partial = True
            self.vprint("Filename = " + self.Filename)
            self.FF = open(Filename, "a", 1)
            self.vprint("Saving:")
            self.vprint("NParam, iter, N init, NFront,"
                        "GenDist, SS, HV, HausDist, Cover, GDPS, SSPS,"
                        "HDPS, NewProb, q, FrontFilename")
        else:
            self.__save_partial = False

        if kernel is None:
            kernel = Matern(nu=1.5)

        self.GP = [None] * self.NObj
        self.Y_MAX = np.array([-np.inf] * self.NObj)

        for i in range(self.NObj):
            self.GP[i] = GPR(kernel=kernel,
                             n_restarts_optimizer=self.n_rest_opt)

        # store starting points
        self.init_points = []

        # test for constraint types
        for cc in self.constraints:
            if cc['type'] == 'eq':
                raise ConstraintError(
                    "Equality constraints are not implemented")

        self.space = NoTargetSpace(self.NObj, self.pbounds,
                                   self.constraints,
                                   RandomSeed=RandomSeed,
                                   verbose=self.verbose)

        if self.Picture and self.NObj == 2:
            self.fig, self.ax = pl.subplots(1, 1, figsize=(5, 4))
            self.fig.show()

        self.early_stop = False
        self.ml_regressor = ml_regressor

        return

    def initialize(self, init_points=None, Points=None, Y=None):

        self.N_init_points = 0
        if Points is not None:
            if Y is None:
                for x in Points:
                    self.space.observe_point(np.array(x))
                    self.N_init_points += 1
            else:
                for x, y in zip(Points, Y):
                    self.space.add_observation(np.array(x), np.array(y))
                    self.N_init_points += 1

        if self.N_init_points == 0:
            raise RuntimeError(
                "A non-zero number of initialization points is required")

        self.vprint("Added points in init")
        self.vprint(self.space.x)

        self.__CalledInit = True
        if len(Points) > 10:
            random_index = np.random.choice(range(len(Points)), size=10, replace=False)
            self.x_init = np.array(Points)[random_index]
        else:
            self.x_init = np.array(Points)

        self.Y_MAX = Y.max(axis=0)
        return

    def maximize(self,
                 n_iter=100,
                 prob=0.1,
                 ReduceProb=False,
                 q=0.5,
                 n_pts=100,
                 SaveInterval=10,
                 FrontSampling=[10, 25, 50, 100]):

        if not self.__CalledInit:
            raise RuntimeError("Initialize was not called, "
                               "call it before calling maximize")

        if not isinstance(n_iter, int):
            raise TypeError(f"n_iter should be int, {type(n_iter)} instead")

        if not isinstance(n_pts, int):
            raise TypeError(f"n_pts should be int, "
                            f"{type(n_pts)} instead")

        if not isinstance(SaveInterval, int):
            raise TypeError(f"SaveInterval should be int, "
                            f"{type(SaveInterval)} instead")

        if isinstance(FrontSampling, list):
            if not all([isinstance(n, int) for n in FrontSampling]):
                raise TypeError(f"FrontSampling should be list of int")
        else:
            raise TypeError(f"FrontSampling should be a list")

        if not isinstance(prob, (int, float)):
            raise TypeError(f"prob should be float, "
                            f"{type(prob)} instead")

        if not isinstance(q, (int, float)):
            raise TypeError(f"q should be float, "
                            f"{type(q)} instead")

        if not isinstance(ReduceProb, bool):
            raise TypeError(f"ReduceProb should be bool, "
                            f"{type(ReduceProb)} instead")

        # Allocate necessary space
        if self.N_init_points + n_iter > self.space._n_alloc_rows:
            self.space._allocate(self.N_init_points + n_iter)

        self.q = q
        self.NewProb = prob

        self.vprint("Start optimization loop")

        for i in range(n_iter):

            self.vprint(i, " of ", n_iter)
            if ReduceProb:
                self.NewProb = prob * (1.0 - self.counter / n_iter)

            for i in range(self.NObj):
                yy = self.space.f[:, i]
                self.GP[i].fit(self.space.x, yy)

            pop, logbook, front = NSGAII(self.NObj,
                                         self.__ObjectiveGP,
                                         self.pbounds,
                                         MU=n_pts)

            Population = np.asarray(pop)
            IndexF, FatorF = self.__LargestOfLeast(front, self.space.f)
            IndexPop, FatorPop = self.__LargestOfLeast(Population,
                                                       self.space.x)

            Fator = self.q * FatorF + (1 - self.q) * FatorPop
            Index_try = np.argmax(Fator)

            self.vprint("IF = ", IndexF,
                        " IP = ", IndexPop,
                        " Try = ", Index_try)

            self.vprint("Front at = ", -front[Index_try])

            self.x_try = Population[Index_try]

            if self.Picture:
                plot_1dgp(fig=self.fig, ax=self.ax, space=self.space,
                          iterations=self.counter + len(self.init_points),
                          Front=front, last=Index_try)

            if self.space.RS.uniform() < self.NewProb:

                if self.NParam > 1:
                    ii = self.space.RS.randint(low=0, high=self.NParam - 1)
                else:
                    ii = 0

                self.x_try[ii] = self.space.RS.uniform(
                    low=self.pbounds[ii][0],
                    high=self.pbounds[ii][1])

                self.vprint("Random Point at ", ii, " coordinate")

            dummy = self.space.observe_point(self.x_try)  # noqa

            self.y_Pareto, self.x_Pareto = self.space.ParetoSet()
            self.counter += 1

            self.vprint(f"|PF| = {self.space.ParetoSize:4d} at"
                        f" {self.counter:4d}"
                        f" of {n_iter:4d}, w/ r = {self.NewProb:4.2f}")

            if self.__save_partial:
                for NFront in FrontSampling:
                    if (self.counter % SaveInterval == 0) and \
                            (NFront == FrontSampling[-1]):
                        SaveFile = True
                    else:
                        SaveFile = False
                    Ind = self.space.RS.choice(front.shape[0], NFront,
                                               replace=False)
                    PopInd = [pop[i] for i in Ind]
                    self.__PrintOutput(front[Ind, :], PopInd,
                                       SaveFile)

        return front, np.asarray(pop)

    def maximize_step(self,
                      prob=0.1,
                      ReduceProb=False,
                      q=0.5,
                      n_pts=100,
                      SaveInterval=10,
                      FrontSampling=[10, 25, 50, 100],
                      i_iter=1,
                      n_sample=1):
        # If initialize was not called, call it and allocate necessary space
        if not self.__CalledInit:
            raise RuntimeError("Initialize was not called, "
                               "call it before calling maximize")

        if not isinstance(n_pts, int):
            raise TypeError(f"n_pts should be int, "
                            f"{type(n_pts)} instead")

        if not isinstance(SaveInterval, int):
            raise TypeError(f"SaveInterval should be int, "
                            f"{type(SaveInterval)} instead")

        if isinstance(FrontSampling, list):
            if not all([isinstance(n, int) for n in FrontSampling]):
                raise TypeError(f"FrontSampling should be list of int")
        else:
            raise TypeError(f"FrontSampling should be a list")

        if not isinstance(prob, (int, float)):
            raise TypeError(f"prob should be float, "
                            f"{type(prob)} instead")

        if not isinstance(q, (int, float)):
            raise TypeError(f"q should be float, "
                            f"{type(q)} instead")

        if not isinstance(ReduceProb, bool):
            raise TypeError(f"ReduceProb should be bool, "
                            f"{type(ReduceProb)} instead")

        self.q = q
        self.NewProb = prob

        if ReduceProb:
            self.NewProb = prob * (1.0 - self.counter / i_iter)

        for i in range(self.NObj):
            yy = self.space.f[:, i]
            self.GP[i].fit(self.space.x, yy)

        pop, logbook, front = NSGAII(self.NObj,
                                     self.__ObjectiveGP_acq,
                                     self.pbounds,
                                     MU=n_pts)

        Population = np.asarray(pop)
        IndexF, FatorF = self.__LargestOfLeast(front, self.space.f)
        IndexPop, FatorPop = self.__LargestOfLeast(Population,
                                                   self.space.x)

        Fator = self.q * FatorF + (1 - self.q) * FatorPop

        Index_try = np.argmax(Fator)

        x_try = sample_maximal_distance_points(Population, n_sample, Index_try)

        self.vprint("IF = ", IndexF,
                    " IP = ", IndexPop,
                    " Try = ", Index_try)

        self.vprint("Front at = ", -front[Index_try])

        # x_try = Population[Index_try]

        return x_try

    def __ObjectiveGP(self, x):

        Fator = 1.0e10
        F = [None] * self.NObj
        xx = np.asarray(x).reshape(1, -1)

        Constraints = 0.0
        for cons in self.constraints:
            y = cons['fun'](x)
            if cons['type'] == 'eq':
                Constraints += np.abs(y)
            elif cons['type'] == 'ineq':
                if y < 0:
                    Constraints -= y

        for i in range(self.NObj):
            F[i] = -self.GP[i].predict(xx)[0] + Fator * Constraints

        return F

    def __ObjectiveGP_acq(self, x, kind='ucb'):

        Fator = 1.0e10
        F = [None] * self.NObj
        xx = np.asarray(x).reshape(1, -1)

        Constraints = 0.0
        for cons in self.constraints:
            y = cons['fun'](x)
            if cons['type'] == 'eq':
                Constraints += np.abs(y)
            elif cons['type'] == 'ineq':
                if y < 0:
                    Constraints -= y

        for i in range(self.NObj):
            F[i] = -self.utility(xx, self.GP[i], self.Y_MAX[i], self.ml_regressor[i], kind=kind)[0] + Fator * Constraints

        return F

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

    def utility_update_params(self,
                              _iters_counter,
                              early_stop=False,
                              ):
        if early_stop:
            self.early_stop = True

        if self.early_stop:
            self._regression_decay_rate = 0
        else:
            self._regression_decay_rate = 1.0

    def utility(self, x, gp, y_max, ml_regressor,
                kind='ucb',
                kappa=2.576,
                xi=0.0):

        self._regression_scale_flag = False
        self._regression_decay_rate = 1.0
        result = 0
        if kind == 'ucb':
            result = self._ucb(x, gp, kappa)
            self._regression_scale_flag = True
            self._regression_scale_rate = 1
            if ml_regressor is None:
                return result

        if kind == 'ei':
            result = self._ei(x, gp, y_max, xi)
            if ml_regressor is None:
                return result
            if not self._regression_scale_flag:
                a = 0
                b = 0
                for i in self.x_init:
                    i = i.reshape(1, -1)
                    last_column = 1 - np.sum(i, axis=1)
                    i_augmented = np.stack((x, last_column.reshape(-1, 1)))
                    a += self._ei(i, gp, y_max, xi)[0]
                    b += ml_regressor.predict(i_augmented)[0]

                self._regression_scale_flag = True
                self._regression_scale_rate = a/b

        if kind == 'poi':
            result = self._poi(x, gp, y_max, xi)
            if ml_regressor is None:
                return result
            if not self._regression_scale_flag:
                a = 0
                b = 0
                for i in self.x_init:
                    i = i.reshape(1, -1)
                    last_column = 1 - np.sum(i, axis=1)
                    i_augmented = np.stack((x, last_column.reshape(-1, 1)))
                    a += self._poi(i, gp, y_max, xi)[0]
                    b += ml_regressor.predict(i_augmented)[0]

                self._regression_scale_flag = True
                self._regression_scale_rate = a / b

        last_column = 1 - np.sum(x, axis=1)
        x_augmented = np.hstack((x, last_column.reshape(-1, 1)))
        return result + ml_regressor.predict(x_augmented) * self._regression_decay_rate * self._regression_scale_rate

    def register(self, x, y):
        self.space.add_observation(x, y)

    def __LargestOfLeast(self, front, F):
        NF = len(front)
        MinDist = np.empty(NF)
        for i in range(NF):
            MinDist[i] = self.__MinimalDistance(-front[i], F)

        ArgMax = np.argmax(MinDist)

        Mean = MinDist.mean()
        Std = np.std(MinDist)
        return ArgMax, (MinDist-Mean)/(Std)

    @staticmethod
    def __MinimalDistance(X, Y):
        N = len(X)
        Npts = len(Y)
        DistMin = float('inf')
        for i in range(Npts):
            Dist = 0.
            for j in range(N):
                Dist += (X[j] - Y[i, j]) ** 2
            Dist = np.sqrt(Dist)
            if Dist < DistMin:
                DistMin = Dist
        return DistMin