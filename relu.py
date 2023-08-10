import numpy as np


def relu(input: np.array) -> np.array:
    '''
    Implementation of ReLU function: f(x) = max(0, x).

    :param input: ReLU function input
    :return: ReLU function result
    '''

    return np.maximum(0, input)


def relu_derivative(input: np.array) -> np.array:
    '''
    Derivative of the ReLU function: f'(x) = 0 for x <= 0, 1 for x > 0.

    :param input: function input
    :return: function result
    '''

    return (input > 0).astype(int)
