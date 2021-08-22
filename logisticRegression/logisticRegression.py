import numpy as np
import pandas as pd


def sigmoid(Z):
    """
    Sigmoid function used in logistic regression.

    Equation:
    result = 1 / (1 + e^(-z))
    m - size of training set

    Z - input, target variable (size m x 1)

    return - result of sigmoid function
    """
    z = np.exp(-Z)
    sig = 1 / (1 + z)
    return sig


if __name__ == "__main__":
    # test sigmoid funciton
    arr = np.array([[-1, 0, 1]])
    sig = sigmoid(arr)
    print('Solution for sigmoid function for arr = [-1, 0, 1]:')
    print(' [{0}, {1}, {2}]'.format(sig[0][0], sig[0][1], sig[0][2]))
