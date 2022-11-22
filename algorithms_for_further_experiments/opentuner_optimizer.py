"""
This code is re-written following the structure from bbo-challenge kit repository:
https://github.com/rdturnermtl/bbo_challenge_starter_kit
"""


import opentuner.tuningrunmain
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface as DMI
from opentuner.resultsdb.models import DesiredResult, Result
from opentuner.search.manipulator import (
    ConfigurationManipulator,
    EnumParameter,
    FloatParameter,
    IntegerParameter,
    LogFloatParameter,
    LogIntegerParameter,
    ScaledNumericParameter,
)
from .base_optimizer import BaseOptimizer

import warnings
from argparse import Namespace

DEFAULT_TECHNIQUES = ("AUCBanditMetaTechniqueA",)
MEMORY_ONLY_DB = "sqlite://"

# Monkey patch here! Opentuner is messed up, TuningRunMain changes global log
# settings. We should file in issue report here and have them fix it.
opentuner.tuningrunmain.init_logging = lambda: None
from algorithms_for_further_experiments.utils import clip_chk


def ClippedParam(cls, epsilon=1e-5):
    """Build wrapper class of opentuner parameter class that use clip check to
    keep parameters in the allowed range despite numerical errors.

    Class built on `ScaledNumericParameter` abstract class defined in:
    `opentuner.search.manipulator.ScaledNumericParameter`.

    Parameters
    ----------
    cls : ScaledNumericParameter
        Opentuner parameter class, such as `LogFloatParameter` or
        `FloatParameter`, which transforms the domain of parameter.

    Returns
    -------
    StableClass : ScaledNumericParameter
        New class equivalent to original `cls` but it overwrites the orginal
        `_unscale` method to enforce a clip check to keep the parameters within
        their allowed range.
    """
    assert issubclass(
        cls, ScaledNumericParameter
    ), "this class cls should inherit from the ScaledNumericParameter class"

    class StableClass(cls):
        def _unscale(self, v):
            unscaled_v = super(StableClass, self)._unscale(v)
            unscaled_v = clip_chk(unscaled_v, self.min_value, self.max_value)
            return unscaled_v

    return StableClass


class OpentunerOptimizer(BaseOptimizer):

    def __init__(self, space, techniques=DEFAULT_TECHNIQUES, n_suggestions=1):

        # Opentuner requires DesiredResult to reference suggestion when making
        # its observation. x_to_dr maps the dict suggestion to DesiredResult.
        self.x_to_dr = {}
        # Keep last suggested x and repeat it whenever opentuner gives up.
        self.dummy_suggest = None
        args = Namespace(
            bail_threshold=500,
            database=MEMORY_ONLY_DB,
            display_frequency=10,
            generate_bandit_technique=False,
            label=None,
            list_techniques=False,
            machine_class=None,
            no_dups=False,
            parallel_compile=False,
            parallelism=n_suggestions,
            pipelining=0,
            print_params=False,
            print_search_space_size=False,
            quiet=False,
            results_log=None,
            results_log_details=None,
            seed_configuration=[],
            stop_after=None,
            technique=techniques,
            test_limit=5000,
        )

        # Setup some dummy classes required by opentuner to actually run.
        manipulator = OpentunerOptimizer.build_manipulator(space)
        interface = DMI(args=args, manipulator=manipulator)
        self.api = TuningRunManager(interface, args)
        super(OpentunerOptimizer, self).__init__(space, observe_dict=True, use_init_points=False)

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
        hashable_object = frozenset(d.items())
        return hashable_object

    @staticmethod
    def build_manipulator(space):
        manipulator = ConfigurationManipulator()

        for pname in space:
            pmin, pmax = space[pname]
            ot_param = FloatParameter(pname, pmin, pmax)
            manipulator.add_parameter(ot_param)
        return manipulator

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

        # Update the n_suggestions if it is different from the current setting.
        if self.api.search_driver.args.parallelism != n_suggestions:
            self.api.search_driver.args.parallelism = n_suggestions
            warnings.warn("n_suggestions changed across suggest calls")

        # Require the user to already observe all previous suggestions.
        # Otherwise, opentuner will just recycle old suggestions.
        assert len(self.x_to_dr) == 0, "all the previous suggestions should have been observed by now"

        # The real meat of suggest from opentuner: Get next `n_suggestions`
        # unique suggestions.
        desired_results = [self.api.get_next_desired_result() for _ in range(n_suggestions)]

        # Save DesiredResult object in dict since observe will need it.
        X = []
        using_dummy_suggest = False
        for ii in range(n_suggestions):
            # Opentuner can give up, but the API requires guessing forever.
            if desired_results[ii] is None:
                assert self.dummy_suggest is not None, "opentuner gave up on the first call!"
                # Use the dummy suggestion in this case.
                X.append(self.dummy_suggest)
                using_dummy_suggest = True
                continue

            # Get the simple dict equivalent to suggestion.
            x_guess = desired_results[ii].configuration.data
            X.append(x_guess)

            # Now save the desired result for future use in observe.
            x_guess_ = OpentunerOptimizer.hashable_dict(x_guess)
            assert x_guess_ not in self.x_to_dr, "the suggestions should not already be in the x_to_dr dict"
            self.x_to_dr[x_guess_] = desired_results[ii]
            # This will also catch None from opentuner.
            assert isinstance(self.x_to_dr[x_guess_], DesiredResult)

        assert len(X) == n_suggestions, "incorrect number of suggestions provided by opentuner"
        # Log suggestion for repeating if opentuner gives up next time. We can
        # only do this when it is not already being used since it we will be
        # checking guesses against dummy_suggest in observe.
        if not using_dummy_suggest:
            self.dummy_suggest = X[-1]
        return X

    def observe(self, X, y):
        """Feed the observations back to opentuner.

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
            x_guess_ = OpentunerOptimizer.hashable_dict(x_guess)
            y_ = float(y_)

            # If we can't find the dr object then it must be the dummy guess.
            if x_guess_ not in self.x_to_dr:
                assert x_guess == self.dummy_suggest, "Appears to be guess that did not originate from suggest"
                continue

            # Get the corresponding DesiredResult object.
            dr = self.x_to_dr.pop(x_guess_, None)
            # This will also catch None from opentuner.
            assert isinstance(dr, DesiredResult), "DesiredResult object not available in x_to_dr"

            # Opentuner's arg names assume we are minimizing execution time.
            # So, if we want to minimize we have to pretend y is a 'time'.
            result = Result(time=y_)
            self.api.report_result(dr, result)
