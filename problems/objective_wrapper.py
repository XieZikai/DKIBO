"""
This is a wrapper for objective functions. Objective functions should follow this structure:
-Input parameter: dict
-Output:
"""
import pandas as pd


class ObjectiveWrapper:

    def __init__(self, black_box_function,
                 input_length: int = None,
                 input_columns: dict = None,
                 bound=None):

        self.black_box_function = black_box_function
        assert not (input_length is None and input_columns is None), 'Must provide input length or input column names!'
        # assert input_length == len(input_columns), 'Input length must match your input columns!'

        if input_columns is None:
            input_columns = []
            for i in range(input_length):
                input_columns.append('x'+str(i))

        self.input_columns = input_columns

    def __call__(self, *args, **kwargs):

        if args != ():
            if 'predict' not in dir(self.black_box_function):
                return self.black_box_function(*args)
            else:
                to_predict = pd.DataFrame(args[0])
                if to_predict.shape[-1] == 1:
                    to_predict = to_predict.T
                to_predict.columns = self.input_columns
                return list(self.black_box_function.predict(to_predict))

        input_data = list(kwargs.values())
        if 'predict' not in dir(self.black_box_function):
            return self.black_box_function(*input_data)
        else:
            # input_data = [[i] for i in input_data]
            # print(input_data)
            to_predict = pd.DataFrame(input_data).T
            to_predict.columns = self.input_columns
            return list(self.black_box_function.predict(to_predict))[0]

