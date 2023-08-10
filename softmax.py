import numpy as np


class Softmax:
    '''
    A class that implements a softmax function.
    '''

    def forward(self, input: np.array) -> np.array:
        '''
        Forward pass through the softmax layer.

        :param input: softmax inputs (previous layers' outputs)
        :return: result of softmax function
        '''

        # numerically stable version of softmax function
        e_yi = np.exp(input - np.max(input))
        s = e_yi / (e_yi.sum(axis=1, keepdims=True) + 1e-10)

        # save results for backpropagation
        self.s = s

        return s

    def backward(self) -> np.array:
        '''
        Backward pass through the softmax layer.

        :return: derivative of the loss function with respect to the inputs
        '''

        # derivative of the loss function with respect to the inputs
        delta_e_x = []

        for k in range(self.s.shape[0]):
            s_reshaped = self.s[k].reshape(-1, 1)

            # set si on diagonal, the rest is zero
            # then subtract si*sj
            jacobian = np.diagflat(s_reshaped) - np.dot(s_reshaped, np.transpose(s_reshaped))

            delta_e_x.append(jacobian)

        return np.array(delta_e_x)
