import mobopt as mo
import numpy as np


def f1(x):
    return np.cos(x)


def f2(x):
    return np.sin(x)


def objective(x):
    """ Objective functions to be optimized

    Input: x --> 1-D np.array with NParam elements
    """
    return np.array([f1(x), f2(x)])


Optimizer = mo.MOBayesianOpt(target=objective,
                             NObj=2,
                             pbounds=np.array([[0, 3], [0, 3]]),
                             verbose=True)

Optimizer.initialize(init_points=2)

front, pop = Optimizer.maximize(n_iter=5)

print('front: ', len(front))
print('pop: ', pop)
print('evaluate:', objective(pop))
