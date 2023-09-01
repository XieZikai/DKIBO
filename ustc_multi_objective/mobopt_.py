from mobopt import MOBayesianOpt
from mobopt._bayes import ConstraintError
from mobopt._target_space import TargetSpace
from mobopt._wrapper import GaussianProcessWrapper as GPR
import numpy as np

from warnings import warn
import matplotlib.pyplot as pl
from sklearn.gaussian_process.kernels import Matern


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
                 kernel=None):

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

        return

    def initialize(self, init_points=None, Points=None, Y=None):
        self.N_init_points = 0
        if init_points is not None:
            self.N_init_points += init_points

            # initialize first points for the gp fit,
            # random points respecting the bounds of the variables.
            rand_points = self.space.random_points(init_points)
            self.init_points.extend(rand_points)
            self.init_points = np.asarray(self.init_points)

            # evaluate target function at all intialization points
            for x in self.init_points:
                self.space.observe_point(x)

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