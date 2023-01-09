import numpy as np


def smooth(variables, k=10):
    """Rolling mean of numeric variables"""
    for key, value in variables.items():
        first_two = value[:2]
        y_padded = np.pad(
                array=value,
                pad_width=(int(k / 2), int(k - 1 - k / 2)),
                mode='edge'
                )
        y = np.convolve(y_padded, np.ones(k) / k, mode='valid')
        y[:2] = first_two
        variables[key] = y

    return variables


def equate_length(variables):
    max_length = 0
    for v in variables.values():
        if type(v) is dict:
            for v_ in v.values():
                max_length = max(max_length, len(v_))
        else:
            max_length = max(max_length, len(v))

    for k, v in variables.items():
        if type(v) is dict:
            for k_, v_ in v.items():
                if len(v_) < max_length:
                    v[k_] = rescale_values(v_, max_length)
        else:
            if len(v) < max_length:
                variables[k] = rescale_values(v, max_length)

    return variables


def rescale_values(y, length):
    x_ref = np.arange(length)
    x_short = np.arange(len(y))
    len_x_ref = len(x_ref)
    interp_x = np.linspace(0, len(y), len_x_ref)
    return np.interp(interp_x, x_short, y)

