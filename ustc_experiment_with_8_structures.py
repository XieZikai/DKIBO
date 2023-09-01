from dkibo import DKIBO, UtilityFunction
import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from scipy.stats import norm
import copy
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import round_result

previous_experiment = pd.read_excel(r'C:\Users\darkn\Desktop\data\POD 1~150.xlsx')

target = previous_experiment.columns[1]
a = previous_experiment[target].apply(lambda x: x.split('   ')).to_list()
for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] = eval(a[i][j])
a = np.array(a)
b = previous_experiment[previous_experiment.columns[0]].to_numpy()

small_indices = np.argpartition(b, 50)[:50]
mask = np.ones(a.shape[0], dtype=bool)
mask[small_indices] = False
a = a[mask, :]
b = np.delete(b, small_indices)

previous_experiment = pd.read_excel(r'C:\Users\darkn\Desktop\data\with_structure\BO_new_.xlsx')
b1 = previous_experiment['KM/mM'].to_numpy()
a1 = previous_experiment[['Mn', 'Fe', 'V', 'Cu', 'Co']]
a1 = np.array(a1)
a = np.concatenate([a, a1])
b = np.concatenate([b, b1])

bound = {
    'Mn': (5, 35),
    'Fe': (5, 10),
    'V': (5, 35),
    'Cu': (5, 35),
    'Co': (5, 35),
}


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Layer1 = nn.Linear(in_features=5, out_features=256)
        self.Layer2 = nn.Linear(in_features=256, out_features=128)
        self.Layer3 = nn.Linear(in_features=128, out_features=8)

    def forward(self, x):
        x = F.relu(self.Layer1(x))
        x = F.relu(self.Layer2(x))
        x = self.Layer3(x)

        return x


structure_model = torch.load('POD-ml-premodel.pkl')
structure_model.eval()


class NNWrapper:
    def __init__(self, nn_model: Module):
        self._input_model = nn_model
        self._output_model = RandomForestRegressor(n_estimators=50, max_depth=5)

    def fit(self, X, y):
        """Fit internal neural network model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        x_tensor = torch.Tensor(X)
        x_structure = self._input_model(x_tensor).detach().numpy()
        self._output_model.fit(x_structure, y)

    def predict(self, X):
        return self._output_model.predict(X)


nn_wrapper = NNWrapper(structure_model)

bo = DKIBO(f=None,
           pbounds=bound,
           ml_regressor=nn_wrapper)

for i in range(len(a)):
    # bo.register(line[1:], line[0])
    bo.register(a[i], -b[i])


class NestedUtilityFunction(UtilityFunction):
    def __init__(self, kind, kappa, xi, gp2, kappa_decay=1, kappa_decay_delay=0, ml_regressor=None, x_init=None,
                 max_iter=100):
        super().__init__(kind, kappa, xi, kappa_decay, kappa_decay_delay, ml_regressor, x_init, max_iter)
        self.model = structure_model
        self.gp2 = gp2

    def _ucb(self, x, gp, kappa):
        gp2_input = self.model(torch.Tensor(x)).detach().numpy()
        mean2, std2 = self.gp2.predict(gp2_input, return_std=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            # print('mean, std: ', mean, std)

        return mean + kappa * std + mean2 + kappa * std2

    def _ei(self, x, gp, y_max, xi):
        gp2_input = self.model(torch.Tensor(x)).detach().numpy()
        mean2, std2 = self.gp2.predict(gp2_input, return_std=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        a2 = (mean2 - y_max - xi)
        z2 = a2 / std2
        return a * norm.cdf(z) + std * norm.pdf(z) + a2 * norm.cdf(z2) + std2 * norm.pdf(z2)

    def _poi(self, x, gp, y_max, xi):
        gp2_input = self.model(torch.Tensor(x)).detach().numpy()
        mean2, std2 = self.gp2.predict(gp2_input, return_std=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        z2 = (mean2 - y_max - xi) / std2
        return norm.cdf(z) + norm.cdf(z2)


constraint = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] - 100})

BATCH = 8
RESAMPLE = 1

all_results = []
x_results = []
i = 0

gp2_x = structure_model(torch.Tensor(a)).detach().numpy()
gp2 = GaussianProcessRegressor(
    kernel=Matern(nu=2.5),
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=5,
)
gp2.fit(gp2_x, b)

x_init = copy.deepcopy(bo._queue._queue)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while i < BATCH:
        k = np.random.uniform(0, 10)
        util = UtilityFunction(kind='ucb',
                               kappa=k,
                               xi=0.0,
                               kappa_decay=1,
                               kappa_decay_delay=0,
                               ml_regressor=None,
                               x_init=x_init,
                               max_iter=100,
                               # gp2=gp2,
                               )
        result_list = []
        x_probe_list = []
        for _ in range(RESAMPLE):
            x_probe = bo.suggest(util, constraint, n_iter=50)
            x_probe_list.append(x_probe)
            result_list.append(
                util.utility(np.array(list(x_probe.values())).reshape(1, -1), bo._gp, bo._space.target.max())[0])
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
            print(k, mean, std)

            # bo.register(x, mean[0])

            x_results.append(round_result(x_probe, constraint))
            all_results.append((round_result(x_probe, constraint), mean, std))
            i += 1
print(all_results)
