import numpy as np


def round_result(x, constraint=None):

    dict_keys = list(x.keys())
    dict_array = np.array(list(x.values()))
    dict_array_rounded = dict_array.round()

    def array_to_dict():
        rounded_dict = {}
        for i in range(len(dict_keys)):
            rounded_dict[dict_keys[i]] = dict_array_rounded[i]
        return rounded_dict

    if constraint is None:
        return array_to_dict()

    if constraint['type'] == 'eq':
        while constraint['fun'](dict_array_rounded) != 0:
            if constraint['fun'](dict_array_rounded) > 0:
                i = np.argmax(dict_array_rounded - dict_array)
                dict_array_rounded[i] -= 1
            if constraint['fun'](dict_array_rounded) < 0:
                i = np.argmin(dict_array_rounded - dict_array)
                dict_array_rounded[i] += 1
        return array_to_dict()

    if constraint['type'] == 'ineq':
        while constraint['fun'](dict_array_rounded) > 0:
            i = np.argmax(dict_array_rounded - dict_array)
            dict_array_rounded[i] -= 1
        return array_to_dict()
