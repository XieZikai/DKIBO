from .objective_wrapper import ObjectiveWrapper

from .data.baseline_model import *
from .data.data_preprocessing import *


def get_kuka_problem():

    dataset = get_kuka_data()
    label_column = 'target'
    predictor_model = get_baseline_model()
    input_columns = []
    input_length = 0
    for column in dataset:
        if column != label_column:
            input_length += 1
            input_columns += [column]

    kuka_problem = ObjectiveWrapper(input_columns=input_columns, black_box_function=predictor_model)

    return kuka_problem, input_columns


if __name__ == '__main__':
    k, input_columns = get_kuka_problem()
    input_dict = {}
    for i in input_columns:
        input_dict[i] = 1

    print(k(**input_dict))