from experiment_results_for_bo_variants.gp_with_mean import BOWithMean
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from dkibo import DKIBO
from problems.photocatalysis_problems import *
from problems.standard_test_problems import *
from problems.gym_problems import *
from algorithms_for_further_experiments import *
from experiments_for_test_bos import check_model_test_problem
from experiments_other_algorithms import check_algorithm_test_problem

import time
import os

test_problems = [
    Colville,
    Michalewicz,
    Ackley,
    Branin,
    Eggholder,
    GoldsteinPrice,
    Hartmann6,
    Rosenbrock,
    SixHumpCamel,
    StyblinskiTang
]

photocatalysis_problem, input_columns = get_kuka_problem()
bound = {}
for i in input_columns:
    bound[i] = (0, 5)
photocatalysis_problem.bound = bound
photocatalysis_problem.name = 'photocatalysis_experiment'

swimmer_problem = Swimmer()
bound = {}
for i, column in enumerate(swimmer_problem.input_columns):
    bound[column] = (swimmer_problem.lb[i], swimmer_problem.ub[i])
swimmer_problem = ObjectiveWrapper(swimmer_problem, input_length=len(swimmer_problem.input_columns), input_list=True)
swimmer_problem.bound = bound
swimmer_problem.name = 'swimmer_experiment'

PATH = r'.'

datetime = time.localtime()
folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                      str(datetime.tm_hour), str(datetime.tm_min))
PATH = os.path.join(PATH, folder_name)


def check_model_test_problem(problem, BO, max_iter=50, n_iter=100, save_result=True, acq='ucb',
                             kappa=2.576, ml_regressor=None, use_noise=True, manual_early_stop=None,
                             early_stop_threshold=0.05):
    global regressor
    result_linear_custom = pd.DataFrame([])
    iter = 0
    path = PATH + '_' + acq + '_' + str(kappa)
    if not os.path.exists(path):
        os.makedirs(path)
    while iter < max_iter:
        if ml_regressor == RandomForestRegressor:
            regressor = ml_regressor(n_estimators=20, max_depth=5)
            name = 'RFBO'
        elif ml_regressor == LinearRegression:
            regressor = ml_regressor()
            name = 'LinearBO'
        elif ml_regressor == GradientBoostingRegressor:
            name = 'GBRT'
            regressor = ml_regressor(n_estimators=20)
        else:
            regressor = None
            name = 'standardBO'
        optimizer = BO(f=problem, pbounds=problem.bound, ml_regressor=regressor, use_noise=use_noise, random_state=iter,
                       early_stop_threshold=early_stop_threshold)

        optimizer.maximize(n_iter=n_iter, acq=acq, kappa=kappa, init_points=5, manual_early_stop=manual_early_stop)
        result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe, dtype=np.float64),
                                                           ignore_index=True)
        iter += 1

    if save_result:
        result_linear_custom = pd.DataFrame(np.array(result_linear_custom))
        result_linear_custom.to_csv(
            os.path.join(path, '{}_result_test_{}_{}.csv'.format(name, problem.name, kappa)))


optimizers = [
    RandomOptimizer,
    ScikitOptimizer,
    OpentunerOptimizer,
    HyperoptOptimizer
]


def ablation(problem, configs, model=DKIBO):
    def run_synthetic_experiment_with_config(config):
        check_model_test_problem(problem, model, max_iter=config['max_iter'], kappa=config['kappa'],
                                 acq=config['acq'], ml_regressor=config['regression'])

    for config in configs:
        run_synthetic_experiment_with_config(config)


def check_swimmer_problem_performance():
    check_model_test_problem(swimmer_problem, DKIBO, ml_regressor=RandomForestRegressor, use_noise=True, acq='ucb')
    check_model_test_problem(swimmer_problem, DKIBO, use_noise=True, acq='ucb')
    check_algorithm_test_problem(swimmer_problem, optimizers, path=PATH)
    check_model_test_problem(swimmer_problem, DKIBO, ml_regressor=LinearRegression, use_noise=True, acq='ucb')


def ablation_swimmer():
    configs = [
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 5.152, 'regression': GradientBoostingRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 5.152, 'regression': RandomForestRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 5.152, 'regression': None},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 1.288, 'regression': GradientBoostingRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 1.288, 'regression': RandomForestRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 1.288, 'regression': None},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 2.576, 'regression': GradientBoostingRegressor},
    ]
    ablation(swimmer_problem, configs)


def ablation_syn(prob):
    problem = TestProblem(prob, minimize=True)
    problem.name = prob.__name__
    configs = [
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 5.152, 'regression': GradientBoostingRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 5.152, 'regression': RandomForestRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 5.152, 'regression': None},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 1.288, 'regression': GradientBoostingRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 1.288, 'regression': RandomForestRegressor},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 1.288, 'regression': None},
        {'max_iter': 50, 'acq': 'ucb', 'kappa': 2.576, 'regression': GradientBoostingRegressor},
    ]
    ablation(problem, configs)


def multiprocess_test_synthetic_functions():
    from multiprocessing import Process

    process_list = []
    for test_problem in test_problems:
        p = Process(target=ablation_syn,
                    args=(test_problem, ))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()


if __name__ == '__main__':
    multiprocess_test_synthetic_functions()
    ablation_swimmer()
    check_model_test_problem(photocatalysis_problem, BOWithMean, ml_regressor=LinearRegression, use_noise=True,
                             acq='ucb')
    check_model_test_problem(photocatalysis_problem, BOWithMean, ml_regressor=LinearRegression, use_noise=True,
                             acq='ucb', early_stop_threshold=None)

