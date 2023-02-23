from experiment_new_ablation.gp_with_mean import BOWithMean
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from problems.photocatalysis_problems import *
from problems.standard_test_problems import *
from experiments_for_test_bos import check_model_test_problem

import time
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

PATH = r'.'

datetime = time.localtime()
folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                      str(datetime.tm_hour), str(datetime.tm_min))
PATH = os.path.join(PATH, folder_name)


def check_model_test_problem(problem, BO, max_iter=50, n_iter=100, save_result=True, path=PATH, acq='ei',
                             kappa=2.576, ml_regressor=None, use_noise=True):
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
        else:
            regressor = None
            name = 'standardBO'
        optimizer = BO(f=problem, pbounds=problem.bound, ml_regressor=regressor, use_noise=use_noise, random_state=iter)

        optimizer.maximize(n_iter=n_iter, acq=acq, kappa=kappa, init_points=5)
        result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe, dtype=np.float64),
                                                           ignore_index=True)
        iter += 1
    if save_result:
        result_linear_custom = pd.DataFrame(np.array(result_linear_custom))
        result_linear_custom.to_csv(
            os.path.join(path, '{}_result_test_{}.csv'.format(name, problem.name)))


check_model_test_problem(photocatalysis_problem, BOWithMean, ml_regressor=LinearRegression, use_noise=True, acq='ucb')