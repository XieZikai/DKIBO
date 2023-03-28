from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from dkibo import DKIBO
from problems.photocatalysis_problems import *
from problems.standard_test_problems import *

test_problems = [
    Colville,
    Michalewicz,
    Ackley,
    Branin,
    Eggholder,
    GoldsteinPrice,
    SixHumpCamel,
    Hartmann6,
    Rosenbrock,
    StyblinskiTang
]

photocatalysis_problem, input_columns = get_kuka_problem()
bound = {}
for i in input_columns:
    bound[i] = (0, 5)
photocatalysis_problem.bound = bound
photocatalysis_problem.name = 'photocatalysis_experiment'

path = r'/experiment_results'
import time

datetime = time.localtime()
folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                      str(datetime.tm_hour), str(datetime.tm_min))
path = os.path.join(path, folder_name)


def check_model_test_problem(problem, BO, max_iter=50, n_iter=100, save_result=True, path=path, acq='ei',
                             kappa=2.576, ml_regressor=None, use_noise=True):
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
        optimizer = BO(f=problem, pbounds=problem.bound, ml_regressor=regressor, use_noise=use_noise, random_state=iter)

        optimizer.maximize(n_iter=n_iter, acq=acq, kappa=kappa, init_points=5)
        result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe, dtype=np.float64),
                                                           ignore_index=True)
        iter += 1
        if regressor is not None:
            optimizer.proportion_dataframe.to_csv(
                os.path.join(path, '{}_early_stop_proportion_trial_{}_{}.csv'.format(name, problem.name, iter)))
    if save_result:
        result_linear_custom = pd.DataFrame(np.array(result_linear_custom))
        result_linear_custom.to_csv(
            os.path.join(path, '{}_result_test_{}.csv'.format(name, problem.name)))


def run_all(regression_list=None, kappa=2.576, use_noise=True):
    if regression_list is None:
        vanilla = None
        regression_list = [RandomForestRegressor, vanilla]
    for regression in regression_list:
        multiprocess_test_synthetic_functions(model=regression, use_noise=use_noise, kappa=kappa)
    check_model_test_problem(photocatalysis_problem, DKIBO, use_noise=use_noise, acq='ucb')
    check_model_test_problem(photocatalysis_problem, DKIBO, ml_regressor=LinearRegression, use_noise=use_noise, acq='ucb')


def run_standardBO(use_noise=True, kappa=2.576):
    multiprocess_test_synthetic_functions(model=None, use_noise=use_noise, kappa=kappa)
    check_model_test_problem(photocatalysis_problem, DKIBO, acq='ucb', use_noise=use_noise)


def experiment_process(BO, regression_model, problem, use_noise=True, kappa=2.576, dir_path=path):
    if problem == photocatalysis_problem:
        check_model_test_problem(photocatalysis_problem, BO, acq='ucb', ml_regressor=regression_model, path=dir_path,
                                 use_noise=use_noise, kappa=kappa)
    else:
        test_function = TestProblem(problem, minimize=True)
        check_model_test_problem(test_function, BO, acq='ucb', ml_regressor=regression_model, use_noise=use_noise,
                                 kappa=kappa)


def multiprocess_test_synthetic_functions(optimizer=DKIBO, model=RandomForestRegressor, use_noise=False, **kwargs):
    from multiprocessing import Process

    process_list = []
    for problem in test_problems:
        p = Process(target=experiment_process,
                    args=(optimizer, model, problem, use_noise, kwargs['kappa'], path))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()


if __name__ == '__main__':
    run_all()
