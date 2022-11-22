from olympus.emulators.emulator import Emulator
from olympus import list_datasets
import olympus
import os
import json
import numbers
import numpy as np


class OlympusEmulatorWrapper(object):

    def __init__(self, dataset='alkox', model='BayesNeuralNet'):
        assert dataset in list_datasets(),\
            print('Not supported dataset! All available datasets: \n', list_datasets())
        self.emulator = Emulator(dataset=dataset, model=model)
        self.dataset = dataset
        self.input_columns, self.bound = self.get_names_and_bounds()
        self.goal = 1

    def get_names_and_bounds(self):
        olympus_path = olympus.__file__.strip('\\__init__.py')
        config_path = os.path.join(os.path.join(os.path.join(olympus_path, 'datasets'), 'dataset_' + self.dataset),
                                   'config.json')
        with open(config_path, 'r') as fp:
            config = json.load(fp)
        names = []
        bound = {}
        for i in config['parameters']:
            names += [i['name']]
            if 'low' in i.keys():
                bound[i['name']] = (i['low'], i['high'])
            elif 'lower' in i.keys():
                bound[i['name']] = (i['lower'], i['upper'])
            else:
                raise KeyError('Not found boundary, please check the configuration file')
        if 'goal' in config['measurements'][0].keys() and config['measurements'][0]['goal'] == 'minimize':
            self.goal = -1
        if 'default_goal' in config.keys() and config['default_goal'] == 'minimize':
            self.goal = -1
        return names, bound

    def experiment(self, **kwargs):
        assert len(kwargs) == len(self.input_columns), \
            "Input length {} doesn't match the experiment model ({} required)!" \
                .format(len(kwargs), len(self.input_columns))
        data = []
        for i in self.input_columns:
            assert isinstance(kwargs[i], numbers.Number), "Invalid input element type {} !".format(type(i))
            data += [kwargs[i]]
        return self.emulator.run(data)[0][0] * self.goal


if __name__ == '__main__':
    print(olympus.__file__)
