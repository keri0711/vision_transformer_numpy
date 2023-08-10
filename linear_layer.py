import numpy as np


class LinearLayer:
    '''
    A class that represents one linear layer that can be used in a neural network.
    It implements a linear transformation Y = XW + B.
    '''

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        '''
        Initialize weights and bias with random numbers from a uniform distribution over [0,1>.

        :param input_size: input dimensions are (1, input_size)
        :param output_size: output dimensions are (input_size, output_size)
        '''

        # initialize weights W
        self.weights = np.random.rand(input_size, output_size)

        # initialize bias B
        if bias:
            self.bias = np.random.rand(1, output_size)
        else:
            self.bias = np.zeros((1, output_size))

    def forward(self, input: np.array) -> np.array:
        '''
        Forward pass through the linear layer.

        :param input: inputs of shape (1, input_size)
        :return: a linear transformation Y = XW + B
        '''

        # save input for backpropagation
        self.input = input

        # calculate Y = XW + B
        y = np.dot(input, self.weights) + self.bias

        return y

    def backward(self, delta_e_y: np.array, learning_rate: float) -> np.array:
        '''
        Backward pass through the linear layer. Used for backpropagation.

        :param delta_e_y: derivative of the loss function with respect to the outputs
        :param learning_rate: learning rate used to update parameters
        :return: derivative of the loss function with respect to the inputs
        '''

        # derivative of the loss function with respect to the weights
        delta_e_w = np.dot(np.transpose(self.input), delta_e_y)

        # derivative of the loss function with respect to the inputs
        delta_e_x = np.dot(delta_e_y, np.transpose(self.weights))

        # update weights according to the delta rule
        self.weights -= learning_rate * delta_e_w
        # update bias according to the delta rule
        # delta_e_b = delta_e_y
        self.bias -= learning_rate * np.mean(delta_e_y, axis=0)

        # derivative of the loss function with respect to the inputs
        # will be used as derivative of the loss function with respect to the outputs
        # in the previous layers
        return delta_e_x
