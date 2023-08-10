import numpy as np
from collections.abc import Callable


class ActivationFunction:
    '''
    A class representing an activation function used in neural networks to achieve non-linearity.
    '''

    def __init__(self, f: Callable[[np.array], np.array], f_derivative: Callable[[np.array], np.array]):
        self.f = f
        self.f_derivative = f_derivative

    def forward(self, input: np.array) -> np.array:
        '''
        Forward pass through the activation function.

        :param input: activation function input
        :return: activation function result
        '''

        # save input for backpropagation
        self.input = input

        # calculate activation function
        y = self.f(input)

        # return result
        return y

    def backward(self, delta_e_y: np.array) -> np.array:
        '''
        Backward pass through the activation function. Used for backpropagation.

        :param delta_e_y: derivative of the loss function with respect to the outputs
        :return: derivative of the loss function with respect to the inputs
        '''

        # derivative of the loss function with respect to the inputs
        delta_e_x = delta_e_y * self.f_derivative(self.input)

        # derivative of the loss function with respect to the inputs
        # will be used as derivative of the loss function with respect to the outputs
        # in the previous layers
        return delta_e_x
