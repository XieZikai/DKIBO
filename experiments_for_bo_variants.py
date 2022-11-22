from dkibo import DKIBO

from problems.standard_test_problems import *
from problems.photocatalysis_problems import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd
import os

test_problems = [
    Ackley,
    Branin,
    Eggholder,
    GoldsteinPrice,
    SixHumpCamel,
    Shekel,
    Hartmann6,
    Michalewicz,
    Rosenbrock,
    StyblinskiTang
]

photocatalysis_problem, input_columns = get_kuka_problem()
bound = {}
for i in input_columns:
    bound[i] = (0, 5)
photocatalysis_problem.bound = bound
photocatalysis_problem.name = 'photocatalysis_experiment'

path = r'./experiment_results_for_bo_variants'
import time

datetime = time.localtime()
folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                      str(datetime.tm_hour), str(datetime.tm_min))
path = os.path.join(path, folder_name)


def check_model_test_problem(problem, BO, max_iter=50, save_result=True, path=path, acq='ucb',
                             kappa=2.576, ml_regressor=None):
    global regressor
    result_linear_custom = pd.DataFrame([])
    iter = 0
    path = path + '_' + acq + '_' + str(kappa)
    if ml_regressor == RandomForestRegressor:
        path = path + '_RF'
    if not os.path.exists(path):
        os.makedirs(path)
    while iter < max_iter:
        if ml_regressor == RandomForestRegressor:
            regressor = ml_regressor(n_estimators=20, max_depth=5)
        elif ml_regressor == LinearRegression:
            regressor = ml_regressor()
        optimizer = BO(f=problem, pbounds=problem.bound, ml_regressor=regressor)

        optimizer.maximize(n_iter=100, acq=acq, kappa=kappa, init_points=5)
        result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe, dtype=np.float64),
                                                           ignore_index=True)
        iter += 1
    if save_result:
        result_linear_custom = pd.DataFrame(np.array(result_linear_custom))
        result_linear_custom.to_csv(os.path.join(path, BO.__name__ + '_result_test_{}.csv'.format(problem.name)))


def run_experiment_with_config(config):
    check_model_test_problem(photocatalysis_problem, DKIBO, max_iter=config['max_iter'], kappa=config['kappa'],
                             acq=config['acq'], ml_regressor=config['regression'])


config_default = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb', 'regression': None}
config_1 = {'max_iter': 50, 'kappa': 5.152, 'acq': 'ucb', 'regression': RandomForestRegressor}
config_2 = {'max_iter': 50, 'kappa': 5.152, 'acq': 'ucb', 'regression': None}
config_3 = {'max_iter': 50, 'kappa': 1.288, 'acq': 'ucb', 'regression': RandomForestRegressor}
config_4 = {'max_iter': 50, 'kappa': 1.288, 'acq': 'ucb', 'regression': None}
config_5 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb_without_mean', 'regression': RandomForestRegressor}
config_6 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'regression_only', 'regression': RandomForestRegressor}
config_7 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb_without_mean', 'regression': LinearRegression}

configs = [config_1, config_2, config_3, config_4, config_5, config_6, config_7]
for config in configs:
    run_experiment_with_config(config)
