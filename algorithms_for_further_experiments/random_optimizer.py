from .base_optimizer import BaseOptimizer
import bayesmark.random_search as rs
import numpy as np


class RandomOptimizer(BaseOptimizer):

    def __init__(self, space):
        super(RandomOptimizer, self).__init__(space)

    def suggest(self, n_suggestions=1):
        x_guess = []
        for _ in range(n_suggestions):
            x_guess.append(self.random_sampling())
        return x_guess

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        pass
