from autogluon.tabular import TabularPredictor
from autogluon.tabular import TabularDataset

from . import data_preprocessing

kuka_data = data_preprocessing.get_kuka_data()


def generate_baseline_model(train_data=kuka_data, label_column='target'):
    save_dir = 'models/'
    data_train_X = TabularDataset(train_data)
    predictor = TabularPredictor(label=label_column).fit(data_train_X)
    return predictor


def get_baseline_model(model_dir=r'C:\Users\darkn\PycharmProjects\ChemicalOptimization\data\AutogluonModels\ag-20211124_140019'):
    return TabularPredictor.load(model_dir)


if __name__ == '__main__':
    _ = generate_baseline_model()
