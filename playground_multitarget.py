from multi_objective_dkibo import MultiObjectiveDKIBO
import numpy as np
import pandas as pd


def target_function(x):
    y0 = np.sum(x) - np.sqrt(x[3]) - np.sqrt(x[2])
    y1 = 5 * np.sin(x[0]) - x[1] ** 2
    y2 = x[0] ** 2 + x[3] ** 2 + np.sqrt(x[2])
    return [y0, y1, y2]


def constraint(x):
    return 0.5 - x[0] -x[1] - x[2] - x[3]


bo = MultiObjectiveDKIBO(
    objective_num=3,
    objective_target=['min', 'min', 'min'],
    pbounds={'n0':(0, 1), 'n1':(0, 1), 'n2':(0, 1), 'n3':(0, 1)},
    ml_regressor=[None, None, None],
    constraint=constraint
)

bo._prime_queue(5)
init_points = bo._queue._queue
for point in init_points:
    bo.register(point, target_function(point))

bo.suggest('ucb', batch_size=8)
