import pandas as pd
import numpy as np
import os

data_dir = r'C:\Users\darkn\PycharmProjects\ChemicalOptimization\data'


def combine_data(data_dir=os.path.join(data_dir, 'further data'), start=0, end=49):
    """
    Combine the data of all .csv file in target path.
    :param data_dir: data storage direction.
    :param start: file start number.
    :param end: file end number.
    :return: combined data of all .csv file
    """
    for i in range(start, end+1):
        assert os.path.exists(os.path.join(data_dir, str(i)+'.csv')), 'file {}.csv is missing'.format(i)
    data = pd.read_csv(os.path.join(data_dir, str(start)+'.csv'), index_col=0)
    for i in range(start+1, end+1):
        new_data = pd.read_csv(os.path.join(data_dir, str(i)+'.csv'), index_col=0)
        data = pd.concat([data, new_data])
    return data


def get_kuka_data(data_dir=data_dir):
    """
    Get the experimental data from 2020 Nature and drop all unuseful information.
    :param data_dir: data storage direction.
    :return:
    """
    data = pd.read_csv(os.path.join(data_dir, 'kuka data.csv'))
    data.rename(columns={data.columns[0]: 'target'}, inplace=True)
    data.drop(columns=data.columns[1], inplace=True)
    # bo_data = data.iloc[:, ]
    exp_data = data.iloc[:, :12]
    return exp_data


