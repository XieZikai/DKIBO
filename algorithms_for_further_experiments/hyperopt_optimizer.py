"""
This code is re-written following the structure from bbo-challenge kit repository:
https://github.com/rdturnermtl/bbo_challenge_starter_kit
"""


import numpy as np
from hyperopt import hp, tpe
from hyperopt.base import JOB_STATE_DONE, JOB_STATE_NEW, STATUS_OK, Domain, Trials
from .utils import random_seed
from .base_optimizer import BaseOptimizer

SEED_MAX_INCL = np.iinfo(np.uint32).max


def dummy_f(x):
    assert False, "This is a placeholder, it should never be called."


class HyperoptOptimizer(BaseOptimizer):

    def __init__(self, space):

        search_space = {'type': 'problem'}
        for key in list(space.keys()):
            search_space[key] = hp.uniform(key, space[key][0], space[key][1])

        self.trial_id_lookup = {}
        search_space = hp.choice('problem',[
            search_space
        ])
        self.domain = Domain(dummy_f, search_space)
        self.trials = Trials()

        super(HyperoptOptimizer, self).__init__(space, use_init_points=False, observe_dict=True)

    def _suggest(self):
        """Helper function to `suggest` that does the work of calling
        `hyperopt` via its dumb API.
        """
        new_ids = self.trials.new_trial_ids(1)
        assert len(new_ids) == 1
        self.trials.refresh()

        seed = random_seed()
        new_trials = tpe.suggest(new_ids, self.domain, self.trials, seed, n_startup_jobs=self.init_points)
        assert len(new_trials) == 1

        self.trials.insert_trial_docs(new_trials)
        self.trials.refresh()

        new_trial, = new_trials  # extract singleton
        return new_trial

    def get_trial(self, trial_id):
        for trial in self.trials._dynamic_trials:
            if trial["tid"] == trial_id:
                assert isinstance(trial, dict)
                # Make sure right kind of dict
                assert "state" in trial and "result" in trial
                assert trial["state"] == JOB_STATE_NEW
                return trial
        assert False, "No matching trial ID"

    @staticmethod
    def hashable_dict(d):
        """A custom function for hashing dictionaries.

        Parameters
        ----------
        d : dict or dict-like
            The dictionary to be converted to immutable/hashable type.

        Returns
        -------
        hashable_object : frozenset of tuple pairs
            Bijective equivalent to dict that can be hashed.
        """
        nd = {}
        for key in d.keys():
            if isinstance(d[key], list):
                nd[key] = d[key][0]
        hashable_object = frozenset(nd.items())
        return hashable_object

    def suggest(self, n_suggestions=1):
        """Make `n_suggestions` suggestions for what to evaluate next.

        This requires the user observe all previous suggestions before calling
        again.

        Parameters
        ----------
        n_suggestions : int
            The number of suggestions to return.

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        assert n_suggestions >= 1, "invalid value for n_suggestions"

        # Get the new trials, it seems hyperopt either uses random search or
        # guesses one at a time anyway, so we might as welll call serially.
        new_trials = [self._suggest() for _ in range(n_suggestions)]

        X = []
        for trial in new_trials:
            x_guess = trial["misc"]["vals"]
            X.append(x_guess)

            # Build lookup to get original trial object
            x_guess_ = HyperoptOptimizer.hashable_dict(x_guess)
            assert x_guess_ not in self.trial_id_lookup, "the suggestions should not already be in the trial dict"
            self.trial_id_lookup[x_guess_] = trial["tid"]

        assert len(X) == n_suggestions
        return X

    @staticmethod
    def dict2array(d):
        # rewritting
        import copy
        nd = copy.deepcopy(d)
        del nd['problem']
        return np.array(list((nd.values()))).T

    def observe(self, X, y):
        """Feed the observations back to hyperopt.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated.
        """
        assert len(X) == len(y)

        for x_guess, y_ in zip(X, y):
            x_guess_ = HyperoptOptimizer.hashable_dict(x_guess)
            assert x_guess_ in self.trial_id_lookup, "Appears to be guess that did not originate from suggest"

            assert x_guess_ in self.trial_id_lookup, "trial object not available in trial dict"
            trial_id = self.trial_id_lookup.pop(x_guess_)
            trial = self.get_trial(trial_id)
            assert trial["misc"]["vals"] == x_guess, "trial ID not consistent with x values stored"

            # Cast to float to ensure native type
            result = {"loss": float(y_), "status": STATUS_OK}
            trial["state"] = JOB_STATE_DONE
            trial["result"] = result
        # hyperopt.fmin.FMinIter.serial_evaluate only does one refresh at end
        # of loop of a bunch of evals, so we will do the same thing here.
        self.trials.refresh()
