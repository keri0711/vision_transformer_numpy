import numpy as np


class SoftmaxCrossEntropy:
    '''
    A class that implements a softmax function followed by cross entropy loss.
    '''

    def forward(self, input: np.array, y_true: np.array) -> np.array:
        '''
        Forward pass through the softmax layer and cross entropy loss.

        :param input: softmax inputs (previous layers' outputs)
        :param y_true: true class labels
        :return: result of softmax and cross entropy loss
        '''

        # softmax function
        # numerically stable version
        e_yi = np.exp(input - np.max(input))
        s = e_yi / (e_yi.sum(axis=1, keepdims=True) + 1e-10)

        # save softmax result for backpropagation
        self.s = s

        # cross entropy loss
        loss = 0.0

        for i in range(len(s)):
            loss -= np.dot(y_true[i], np.log(s[i]))

        # return result
        return loss

    def backward(self, y_true: np.array) -> np.array:
        '''
        Backward pass through the softmax layer and cross entropy loss.

        :param y_true: true class labels
        :return: derivative of the loss function with respect to the inputs
        '''

        # derivative of the loss function with respect to the inputs
        delta_e_x = self.s - y_true

        return delta_e_x
