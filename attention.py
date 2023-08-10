import numpy as np
from softmax import Softmax
from linear_layer import LinearLayer


class Attention:
    '''
    A class that represents one self attention block.
    '''

    def __init__(self, input_size: int):
        '''
        Initialize three linear layers that represent queries, keys and values multiplied by weight matrices.
        Initialize a softmax layer.

        :param input_size: input dimensions are (1, input_size)
        '''

        self.input_size = input_size

        self.linear_layer_1 = LinearLayer(input_size=input_size, output_size=input_size, bias=False)
        self.linear_layer_2 = LinearLayer(input_size=input_size, output_size=input_size, bias=False)
        self.linear_layer_3 = LinearLayer(input_size=input_size, output_size=input_size, bias=False)

        self.softmax_layer = Softmax()

    def forward(self, input: np.array) -> np.array:
        # save input for backpropagation
        self.input = input

        x_w1 = self.linear_layer_1.forward(input=input)
        x_w2 = self.linear_layer_2.forward(input=input)
        x_w3 = self.linear_layer_3.forward(input=input)

        # save it for backpropagation
        self.x_w1 = x_w1
        self.x_w2 = x_w2
        self.x_w3 = x_w3

        softmax_input = np.dot(x_w1, np.transpose(x_w2)) / np.sqrt(self.input_size)
        softmax_input = np.array(softmax_input)

        # save it for backpropagation
        softmax_result = self.softmax_layer.forward(softmax_input)
        self.softmax_result = softmax_result

        attention = np.dot(softmax_result, x_w3)

        return attention

    def backward(self, delta_e_at: np.array, learning_rate: float) -> np.array:
        '''
        Let A = X * W1, B = X * W2, C = X * W3
        D = (A * B^T) / sqrt(input_size)
        S = softmax(D)
        J = jacobian(S)
        Attention = S * C

        delta_e_w1 = (delta_e_at * delta_at_s^T * delta_s_d * delta_d_a)^T * delta_a_w1
                   = (delta_e_at * (X * W3)^T * J * ((X * W2) / sqrt(input_size)))^T * X
        delta_e_w2 = (delta_e_at * delta_at_s^T * delta_s_d * delta_d_b)^T * delta_b_w2
                   = (delta_e_at * (X * W3)^T * J * ((X * W1) / sqrt(input_size)))^T * X
        delta_e_w3 = delta_e_at^T * S * X

        delta_e_x = delta_e_at * delta_at_s^T * delta_s_d * delta_d_a * delta_a_x +
                    delta_e_at * delta_at_s^T * delta_s_d * delta_d_b * delta_b_x +
                    (delta_e_at^T * delta_at_c)^T * delta_c_x
                  = delta_e_at * (X * W3)^T * J * ((X * W2) / sqrt(input_size)) * W1 +
                    delta_e_at * (X * W3)^T * J * ((X * W1) / sqrt(input_size)) * W2 +
                    (delta_e_at^T * S)^T * W3

        :param delta_e_at: derivative of the loss function with respect to the Attention
        :param learning_rate: learning rate used to update parameters
        :return: derivative of the loss function with respect to the inputs
        '''

        # save softmax derivative
        jacobian = self.softmax_layer.backward()

        # W1 gradient
        delta_e_s = np.dot(delta_e_at, np.transpose(self.x_w3))

        delta_e_d = []
        for i in range(jacobian.shape[0]):
            delta_e_d.append(np.dot(delta_e_s[i], jacobian[i]))

        delta_e_a = np.dot(np.array(delta_e_d), self.x_w2 / np.sqrt(self.input_size))
        delta_e_w1 = np.dot(np.transpose(delta_e_a), self.input)

        # update weights according to the delta rule
        self.linear_layer_1.weights -= learning_rate * delta_e_w1

        # W2 gradient
        delta_e_b = np.dot(np.array(delta_e_d), self.x_w1 / np.sqrt(self.input_size))
        delta_e_w2 = np.dot(np.transpose(delta_e_b), self.input)

        # update weights according to the delta rule
        self.linear_layer_2.weights -= learning_rate * delta_e_w2

        # W3 gradient
        delta_e_c = np.dot(np.transpose(delta_e_at), self.softmax_result)
        delta_e_w3 = np.dot(delta_e_c, self.input)

        # update weights according to the delta rule
        self.linear_layer_3.weights -= learning_rate * delta_e_w3

        # X gradient
        delta_e_x_1 = np.dot(delta_e_at, np.transpose(self.x_w3))

        tmp = []
        for i in range(jacobian.shape[0]):
            tmp.append(np.dot(delta_e_x_1[i], jacobian[i]))
        delta_e_x_1 = np.array(tmp)

        delta_e_x_1 = np.dot(delta_e_x_1, self.x_w2 / np.sqrt(self.input_size))
        delta_e_x_1 = np.dot(delta_e_x_1, self.linear_layer_1.weights)

        delta_e_x_2 = np.dot(delta_e_at, np.transpose(self.x_w3))

        tmp = []
        for i in range(jacobian.shape[0]):
            tmp.append(np.dot(delta_e_x_2[i], jacobian[i]))
        delta_e_x_2 = np.array(tmp)

        delta_e_x_2 = np.dot(delta_e_x_2, self.x_w1 / np.sqrt(self.input_size))
        delta_e_x_2 = np.dot(delta_e_x_2, self.linear_layer_2.weights)

        delta_e_x_3 = np.transpose(np.dot(np.transpose(delta_e_at), self.softmax_result))
        delta_e_x_3 = np.dot(delta_e_x_3, self.linear_layer_3.weights)

        delta_e_x = delta_e_x_1 + delta_e_x_2 + delta_e_x_3

        return delta_e_x
