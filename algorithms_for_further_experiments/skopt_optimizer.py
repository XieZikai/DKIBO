"""
This code is re-written following the structure from bbo-challenge kit repository:
https://github.com/rdturnermtl/bbo_challenge_starter_kit
"""


import numpy as np
from scipy.interpolate import interp1d
from skopt import Optimizer as SkOpt
from skopt.space import Categorical, Integer, Real
from .base_optimizer import BaseOptimizer


class ScikitOptimizer(BaseOptimizer):
    name = 'skopt'

    def __init__(self, space, base_estimator="GBRT", acq_func="EI", acq_optimizer="sampling", n_initial_points=5):
        super(ScikitOptimizer, self).__init__(space, observe_dict=False)
        dimensions = ScikitOptimizer.get_sk_dimensions(space)
        self.dimensions_list = tuple(dd.name for dd in dimensions)
        self.skopt = SkOpt(
            dimensions,
            n_initial_points=n_initial_points,
            base_estimator=base_estimator,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            acq_func_kwargs={},
            acq_optimizer_kwargs={},
        )

    @staticmethod
    def get_sk_dimensions(space, transform="normalize"):
        param_list = sorted(space.keys())
        sk_dims = []

        for param_name in param_list:
            sk_dims.append(Real(space[param_name][0], space[param_name][1],
                                prior="uniform",
                                transform=transform,
                                name=param_name))
        return sk_dims

    def suggest(self, n_suggestions=1):
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        # First get list of lists from skopt.ask()
        next_guess = self.skopt.ask(n_points=n_suggestions)
        # Then convert to list of dicts
        next_guess = [dict(zip(self.dimensions_list, x)) for x in next_guess]

        return next_guess

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Supposedly skopt can handle blocks, but not sure about interface for
        # that. Just do loop to be safe for now.
        for xx, yy in zip(X, y):
            if isinstance(xx, dict):
                xx = list(xx.values())
            # skopt needs lists instead of dicts
            # xx = [xx[dim_name] for dim_name in self.dimensions_list]
            # Just ignore, any inf observations we got, unclear if right thing
            if np.isfinite(yy):
                self.skopt.tell(xx, yy)
