from algorithms_for_further_experiments import *
from problems.photocatalysis_problems import *
from problems.standard_test_problems import *
import pandas as pd
import os
import time

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

optimizers = [
    RandomOptimizer,
    ScikitOptimizer,
    OpentunerOptimizer,
    HyperoptOptimizer
]

photocatalysis_problem, input_columns = get_kuka_problem()
bound = {}
for i in input_columns:
    bound[i] = (0, 5)
photocatalysis_problem.bound = bound
photocatalysis_problem.name = 'photocatalysis_experiment'

path = r'/algorithms_for_further_experiments'

datetime = time.localtime()
folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                      str(datetime.tm_hour), str(datetime.tm_min))
path = os.path.join(path, folder_name)


def check_algorithm_test_problem(problem, optimizers, max_iter=50, path=path):
    if not os.path.exists(path):
        os.makedirs(path)

    for optimizer in optimizers:
        iteration = 0
        sub_path = os.path.join(path, optimizer.__name__)
        if not os.path.exists(path):
            os.makedirs(sub_path)

        results = []
        while iteration < max_iter:
            opt = optimizer(problem.bound)
            opt.maximize(problem, n_iter=100)
            results.append(opt.results)
            iteration += 1
        results = np.array(results)
        if len(results.shape) == 3:
            results = results.squeeze()
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(path, '{}_result_test_{}.csv'.format(optimizer.__name__, problem.name)))


if __name__ == '__main__':
    for problem in test_problems:
        test_function = TestProblem(problem, minimize=True)
        check_algorithm_test_problem(test_function, optimizers)
    check_algorithm_test_problem(photocatalysis_problem, optimizers)
