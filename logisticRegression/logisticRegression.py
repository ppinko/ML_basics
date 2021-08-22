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


def hypothesisLogisticRegression(X, theta):
    """
    Calculates hypthothesis for logistic regression.

    Equation:
    h = g(X * theta)
    where:
        g(z) = 1 / (1 + e^(-z)) # sigmoid function

    m - size of training set
    n - number of features (including feature zero - 'bias')

    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)

    return - value based on the hyphothesis
    """
    temp = np.dot(X, theta)
    return sigmoid(temp)


def costLogisticRegression(Y, X, theta):
    """
    Cost function for logistic regression with multiple features.

    Equation:
    cost = (1 / m) * sum((-Y * log(h))-((1 - Y)* log(1 - h)))

    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)

    return - cost for given theta and bias
    """
    m = len(Y)
    leftTrue = -Y * np.log(hypothesisLogisticRegression(X, theta))
    rightFalse = -(1-Y) * np.log(1 - hypothesisLogisticRegression(X, theta))
    return (1.0 / m) * np.add(leftTrue, rightFalse).sum()


def gradientDescentLogisticRegression(Y, X, theta):
    """
    Calculates theta (coefficients for features) using gradient descent
    algorithm.

    Equation:
    grad = (1 / m) * (X' * (h - Y))
    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)

    return - optimal theta (coefficients for features)
    """
    m = len(Y)
    return (1.0 / m) * np.dot(np.transpose(X), h - hypothesisLogisticRegression(X, theta))


if __name__ == "__main__":
    # test sigmoid funciton
    arr = np.array([[-1, 0, 1]])
    sig = sigmoid(arr)
    print('Solution for sigmoid function for arr = [-1, 0, 1]:')
    print(' [{0}, {1}, {2}]'.format(sig[0][0], sig[0][1], sig[0][2]))
