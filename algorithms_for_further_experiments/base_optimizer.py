import numpy as np
import pandas as pd


class BaseOptimizer:
    """
    Base optimizer to wrap 4 baseline optimizers for experiment.
    """
    def __init__(self, space, use_init_points=True, observe_dict=False):
        self.space = space
        self.use_init_points = use_init_points
        self.results = []
        self.observe_dict = observe_dict
        self.init_points = 5  # by default

    def random_sampling(self):
        guess = []
        for key in self.space:
            guess.append(np.random.uniform(low=self.space[key][0], high=self.space[key][1]))
        return guess

    def observe(self, X, y):
        raise NotImplementedError

    def suggest(self, n_suggestions=1):
        raise NotImplementedError

    @staticmethod
    def dict2array(d):
        if not isinstance(d, dict):
            return d
        return list(d.values())

    def dict_or_array(self, d):
        if not self.observe_dict:
            if not isinstance(d, dict):
                return d
            return list(d.values())
        else:
            if not isinstance(d, dict):
                l = {}
                for i, key in enumerate(self.space.keys()):
                    l[key] = d[i]
                return l
            else:
                return d

    def maximize(self, obj, init_points=5, n_iter=100):
        self.init_points = init_points
        best_value = -500000000
        if self.use_init_points:
            print("============================")
            if hasattr(self, 'name'):
                print(self.name)
            print("Initial Guesses: ")
            for i in range(init_points):
                x_guess = self.random_sampling()
                y = obj(self.dict2array(x_guess))
                if isinstance(y, list) or isinstance(y, np.ndarray):
                    y = y[0]
                self.observe([self.dict_or_array(x_guess)], [-y])
                self.results.append(y)
                print("Guess {}, obj value {}".format(i, y))
            print("============================")
        else:
            n_iter += init_points

        iteration = 0
        while iteration < n_iter:
            x_guess = self.suggest(n_suggestions=1)[0]
            y = obj(self.dict2array(x_guess))
            if isinstance(y, list) or isinstance(y, np.ndarray):
                y = y[0]
            self.observe([self.dict_or_array(x_guess)], [-y])
            self.results.append(y)
            if y > best_value:
                best_value = y
                print("Iteration {}, obj value {}, best value reached".format(iteration+1, y))
            else:
                print("Iteration {}, obj value {}".format(iteration + 1, y))
            iteration += 1
