import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from bayes_opt import BayesianOptimization
from dkibo import DKIBO
from problems.photocatalysis_problems import *
from problems.standard_test_problems import *

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
    if not os.path.exists(path):
        os.makedirs(path)
    while iter < max_iter:
        if ml_regressor == RandomForestRegressor:
            regressor = ml_regressor(n_estimators=20, max_depth=5)
            name = 'RFBO'
        elif ml_regressor == LinearRegression:
            regressor = ml_regressor()
            name = 'LinearBO'
        else:
            regressor = None
            name = 'standardBO'
        optimizer = BO(f=problem, pbounds=problem.bound, ml_regressor=regressor, random_state=iter)

        optimizer.maximize(n_iter=100, acq=acq, kappa=kappa, init_points=5)
        result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe, dtype=np.float64),
                                                           ignore_index=True)
        iter += 1
    if save_result:
        result_linear_custom = pd.DataFrame(np.array(result_linear_custom))
        result_linear_custom.to_csv(os.path.join(path, name + '_result_test_{}_{}.csv'.format(problem.name, kappa)))


def photocatalysis():
    def run_photocatalysis_experiment_with_config(config):
        check_model_test_problem(photocatalysis_problem, DKIBO, max_iter=config['max_iter'], kappa=config['kappa'],
                                 acq=config['acq'], ml_regressor=config['regression'])

    config_default = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb', 'regression': None}
    config_1 = {'max_iter': 50, 'kappa': 5.152, 'acq': 'ucb', 'regression': LinearRegression}
    config_2 = {'max_iter': 50, 'kappa': 5.152, 'acq': 'ucb', 'regression': None}
    config_3 = {'max_iter': 50, 'kappa': 1.288, 'acq': 'ucb', 'regression': LinearRegression}
    config_4 = {'max_iter': 50, 'kappa': 1.288, 'acq': 'ucb', 'regression': None}
    config_5 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb_without_mean', 'regression': LinearRegression}
    config_6 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'regression_only', 'regression': LinearRegression}
    # config_7 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb_without_mean', 'regression': LinearRegression}

    configs = [config_1, config_2, config_3, config_4, config_5, config_6]
    for config in configs:
        run_photocatalysis_experiment_with_config(config)


def experiment_process(BO, regression_model, problem, acq='ucb', use_noise=True, kappa=2.576, dir_path=path):
    if problem == photocatalysis_problem:
        check_model_test_problem(photocatalysis_problem, BO, acq='ucb', ml_regressor=regression_model, path=dir_path, kappa=kappa)
    else:
        test_function = TestProblem(problem, minimize=True)
        check_model_test_problem(test_function, BO, acq='ucb', ml_regressor=regression_model, kappa=kappa)


def multiprocess_test_synthetic_functions(optimizer=DKIBO, model=RandomForestRegressor, use_noise=False, **kwargs):
    from multiprocessing import Process

    process_list = []
    for problem in test_problems:
        p = Process(target=experiment_process,
                    args=(optimizer, model, problem, kwargs['acq'], use_noise, kwargs['kappa'], path))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()


def synthetic():
    def run_synthetic_experiment_with_config(config, problem):
        check_model_test_problem(problem, DKIBO, max_iter=config['max_iter'], kappa=config['kappa'],
                                 acq=config['acq'], ml_regressor=config['regression'])

    config_default = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb', 'regression': None}
    config_1 = {'max_iter': 50, 'kappa': 5.152, 'acq': 'ucb', 'regression': RandomForestRegressor}
    config_2 = {'max_iter': 50, 'kappa': 5.152, 'acq': 'ucb', 'regression': None}
    config_3 = {'max_iter': 50, 'kappa': 1.288, 'acq': 'ucb', 'regression': RandomForestRegressor}
    config_4 = {'max_iter': 50, 'kappa': 1.288, 'acq': 'ucb', 'regression': None}
    config_5 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'ucb_without_mean', 'regression': RandomForestRegressor}
    config_6 = {'max_iter': 50, 'kappa': 2.576, 'acq': 'regression_only', 'regression': RandomForestRegressor}

    configs = [config_default, config_1, config_2, config_3, config_4, config_5, config_6]

    sub_test_problems = [StyblinskiTang]
    for problem in sub_test_problems:
        problem_func = TestProblem(problem, minimize=True)
        for config in configs:
            run_synthetic_experiment_with_config(config, problem_func)


photocatalysis()