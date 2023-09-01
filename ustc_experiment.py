from dkibo import DKIBO, UtilityFunction
import pandas as pd
import numpy as np
import copy
from utils import round_result
import warnings

# previous_experiment = pd.read_excel(r'C:\Users\darkn\Desktop\data\POD-1~51.xlsx')
# previous_experiment.drop(columns=['Unnamed: 6', 'Unnamed: 7'], inplace=True)

previous_experiment = pd.read_excel(r'C:\Users\darkn\Desktop\data\POD 1~150.xlsx')

target = previous_experiment.columns[1]
a = previous_experiment[target].apply(lambda x: x.split('   ')).to_list()
for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] = eval(a[i][j])
a = np.array(a)
b = previous_experiment[previous_experiment.columns[0]].to_numpy()

bound = {
    'Mn': (5, 35),
    'Fe': (5, 35),
    'V': (5, 35),
    'Cu': (5, 35),
    'Co': (5, 35),
}


bo = DKIBO(f=None,
           pbounds=bound,
           ml_regressor=None)

for i in range(len(previous_experiment)):
    line = previous_experiment.iloc[i].to_numpy()
    # bo.register(line[1:], line[0])
    bo.register(a[i], -b[i])

previous_experiment = pd.read_excel(r'C:\Users\darkn\Desktop\data\BO.xlsx')
b = previous_experiment['KM/mM'].to_numpy()
a = previous_experiment[['Mn', 'Fe', 'V', 'Cu', 'Co']]

for i in range(len(previous_experiment)):
    line = a.iloc[i].to_numpy()
    bo.register(line, -b[i])

x_init = copy.deepcopy(bo._queue._queue)

constraint = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] - 100})

BATCH = 8
RESAMPLE = 1

all_results = []
x_results = []
i = 0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while i < BATCH:
        '''noise = np.random.normal()
        while noise > 2.5 or noise < -2.5:
            noise = np.random.normal()'''
        noise = np.random.uniform(0, 10)

        util = UtilityFunction(kind='ucb',
                               # kappa=2.576 + noise,
                               kappa=noise,
                               xi=0.0,
                               kappa_decay=1,
                               kappa_decay_delay=0,
                               ml_regressor=None,
                               x_init=x_init,
                               max_iter=100)

        result_list = []
        x_probe_list = []
        for _ in range(RESAMPLE):
            x_probe = bo.suggest(util, constraint)
            x_probe_list.append(x_probe)
            result_list.append(util.utility(np.array(list(x_probe.values())).reshape(1, -1), bo._gp, bo._space.target.max())[0])
        max_index = np.argmax(result_list)
        x_probe = x_probe_list[max_index]
        x = np.array(list(x_probe.values()))
        mean, std = bo._gp.predict(x.reshape(1, -1), return_std=True)

        if round_result(x_probe, constraint) in x_results:
            continue
        else:
            print(round_result(x_probe, constraint))
            x = np.array(list(x_probe.values()))
            mean, std = bo._gp.predict(x.reshape(1, -1), return_std=True)
            print(mean, std)

            bo.register(x, mean[0])

            x_results.append(round_result(x_probe, constraint))
            all_results.append((round_result(x_probe, constraint), mean, std))
            i += 1

print(all_results)
