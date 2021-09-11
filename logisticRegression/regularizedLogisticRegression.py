import logisticRegression as logReg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op


def visualizeDataInitial1():
    """
    Visualize data for microchips quality assurance from two tests.
    """
    # load the dataset
    data = np.loadtxt('microchipsQuailityAssurance.txt',
                      delimiter=',', skiprows=1)

    X = data[:, 0:2]
    y = data[:, 2]

    positive = np.where(y == 1)
    negative = np.where(y == 0)
    plt.scatter(X[positive, 0], X[positive, 1], marker='o', c='g')
    plt.scatter(X[negative, 0], X[negative, 1], marker='x', c='r')
    plt.xlabel('Test 1 score')
    plt.ylabel('Test 2 score')
    plt.legend(['Good quality', 'Bad quality'])
    plt.title(label='Microchips quality assurance')
    plt.show()


def mapFeature(X, degree):
    '''
    Maps the two input features (X1 and X2) to quadratic features used in the
    regularization exercise.

    m - size of training set
    n - number of features (including feature zero - 'bias')

    X - the independent variables (features) (size m x n), where X[:, 0] = 1
    Y - the highest power of any feature

    return -  a new feature array with more features, comprising of X1, X2,
    X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    '''
    res = X[:, 0]
    for i in range(1, degree + 1):
        for j in range(i+1):
            res = np.column_stack(
                (res, np.multiply(np.power(X[:, 1], i - j), np.power(X[:, 2], j))))
    return res


if __name__ == "__main__":
    # visualize data
    visualizeDataInitial1()
