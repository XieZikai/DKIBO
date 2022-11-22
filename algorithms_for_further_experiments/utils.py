import numpy as np

SEED_MAX_INCL = 2**20 - 1


def random_seed():
    """Draw a random seed compatible with :class:`numpy:numpy.random.RandomState`.

    Parameters
    ----------
    random : :class:`numpy:numpy.random.RandomState`
        Random stream to use to draw the random seed.

    Returns
    -------
    seed : int
        Seed for a new random stream in ``[0, 2**32-1)``.
    """
    # np randint is exclusive on the high value, py randint is inclusive. We
    # must use inclusive limit here to work with both. We are missing one
    # possibility here (2**32-1), but I don't think that matters.
    seed = np.random.randint(0, SEED_MAX_INCL)
    return seed


def isclose_lte(x, y):
    """Check that less than or equal to (lte, ``x <= y``) is approximately true between all elements of `x` and `y`.

    This is similar to :func:`numpy:numpy.allclose` for equality. Shapes of all input variables must be broadcast
    compatible.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        Lower limit in ``<=`` check.
    y : :class:`numpy:numpy.ndarray`
        Upper limit in ``<=`` check.

    Returns
    -------
    lte : bool
        True if ``x <= y`` is approximately true element-wise.
    """
    # Use np.less_equal to ensure always np type consistently
    lte = np.less_equal(x, y) | np.isclose(x, y)
    return lte


def clip_chk(x, lb, ub, allow_nan=False):
    """Clip all element of `x` to be between `lb` and `ub` like :func:`numpy:numpy.clip`, but also check
    :func:`numpy:numpy.isclose`.

    Shapes of all input variables must be broadcast compatible.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        Array containing elements to clip.
    lb : :class:`numpy:numpy.ndarray`
        Lower limit in clip.
    ub : :class:`numpy:numpy.ndarray`
        Upper limit in clip.
    allow_nan : bool
        If true, we allow ``nan`` to be present in `x` without out raising an error.

    Returns
    -------
    x : :class:`numpy:numpy.ndarray`
        An array with the elements of `x`, but where values < `lb` are replaced with `lb`, and those > `ub` with `ub`.
    """
    assert np.all(lb <= ub)  # np.clip does not do this check

    x = np.asarray(x)

    # These are asserts not exceptions since clip_chk most used internally.
    if allow_nan:
        assert np.all(isclose_lte(lb, x) | np.isnan(x))
        assert np.all(isclose_lte(x, ub) | np.isnan(x))
    else:
        assert np.all(isclose_lte(lb, x))
        assert np.all(isclose_lte(x, ub))
    x = np.clip(x, lb, ub)
    return x