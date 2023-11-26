import os

import numpy as np
from utils import round_result

from mobopt_ import NoTargetMOBayesianOpt
from get_model.get_pretrain_model import Model
import torch
from torch.nn import Module
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import cdist


class NNWrapper:
    def __init__(self, nn_model: Module):
        self._input_model = nn_model
        self._input_model.eval()
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


def get_bad_points(data, n_sample):
    pareto_front = []
    for i in range(len(data)):
        if all(np.any(data[i] <= data[j]) for j in range(len(data))):
            pareto_front.append(i)
    pareto_front = np.array(pareto_front)
    distances = cdist(data, data[pareto_front, :])
    min_distances = np.min(distances, axis=1)
    farthest_indices = np.argsort(min_distances)[-n_sample:]
    return farthest_indices


def experiment():
    ml_model = torch.load('./get_model/POD-premodel.pkl')

    rf_km = NNWrapper(ml_model)
    rf_vmax = NNWrapper(ml_model)

    metals = pd.read_excel("./get_model/POD_exp.xlsx", sheet_name="metals")
    metals = metals.to_numpy()[100:]

    # minimize Km value
    km_train_y = pd.read_excel("./get_model/POD_exp.xlsx", sheet_name="km")
    km_train_y = km_train_y.to_numpy()
    km_train_y = (-km_train_y)

    # maximize Vmax value
    vmax_train_y = pd.read_excel("./get_model/POD_exp.xlsx", sheet_name="Vmax")
    vmax_train_y = vmax_train_y.to_numpy()

    data = np.concatenate((km_train_y, vmax_train_y), axis=1)
    indices = get_bad_points(data, 50)

    km_train_y = km_train_y[indices]
    vmax_train_y = vmax_train_y[indices]

    # register all batch data
    batch_files = [i for i in os.listdir('./') if 'batch_' in i]
    for batch in batch_files:
        df_batch = pd.read_excel(batch, sheet_name='Sheet1')
        batch_x = df_batch[['Fe', 'Co', 'Cu', 'Mn', 'V']].to_numpy()
        batch_km = df_batch['KM/mM'].to_numpy().reshape(-1, 1)
        batch_km = (-batch_km)
        batch_vmax = df_batch['Vmax'].to_numpy().reshape(-1, 1)

        metals = np.concatenate((metals, batch_x), axis=0)
        km_train_y = np.concatenate((km_train_y, batch_km), axis=0)
        vmax_train_y = np.concatenate((vmax_train_y, batch_vmax), axis=0)

    rf_km.fit(metals, km_train_y)
    rf_vmax.fit(metals, vmax_train_y)

    constraint = [
        {
            'type': 'ineq',
            'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 0.65
        },
        {
            'type': 'ineq',
            'fun': lambda x: 0.95 - (x[0] + x[1] + x[2] + x[3])
        },
    ]

    optimizer = NoTargetMOBayesianOpt(target=None, NObj=2,
                                      pbounds=np.array([[0.05, 0.35], [0.05, 0.35], [0.05, 0.35], [0.05, 0.35]]),
                                      constraints=constraint, ml_regressor=[rf_km, rf_vmax])
    init_x = metals[:, :4]
    init_y = np.concatenate((km_train_y, vmax_train_y), axis=1)
    optimizer.initialize(Points=init_x, Y=init_y)

    results = optimizer.maximize_step(n_sample=8)

    for result in results:
        result = np.append(result, 1 - np.sum(result))

        round_constraint = {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 100
        }
        result = round_result(np.array(result) * 100, round_constraint)
        print(result)


if __name__ == "__main__":
    experiment()