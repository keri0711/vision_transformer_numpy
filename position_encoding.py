import numpy as np


def position_encoding(num_vectors: int, vector_size: int, n=10000) -> np.array:
    '''
    P(k, 2i) = sin(k / (n^(2i / d)))
    P(k, 2i + 1) = cos(k / (n^(2i / d)))

    :param num_vectors: a number of wanted position encoding vectors
    :param vector_size: position encoding vector size
    :param n: a defined scalar
    :return: position encoding vectors
    '''

    P = np.zeros((num_vectors, vector_size))

    for k in range(num_vectors):
        for i in np.arange(int(vector_size/2)):
            denominator = np.power(n, (2*i)/vector_size)

            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)

    return P
