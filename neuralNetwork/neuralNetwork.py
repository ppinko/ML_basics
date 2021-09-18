import typing_extensions
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.io
import random
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plotExamples(cmVal, data):
    """
    Plot examples of hand-writting numbers.

    cmVal - color map
    data - array representing hand-writting numbers using 0..1 value, different
        shades of a coulour (size 100 x 400)

    return - print figure showing the handwritten numbers
    """
    result = np.empty(shape=(200, 200))
    side = 20
    # initiate subplots
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(
        25, 25), constrained_layout=True)

    for i in range(10):
        for j in range(10):
            # prepare a sub-matrix for a single number
            indx = 10 * i + j
            subMatrix = data[indx, :]
            subMatrix = subMatrix.reshape((side, side))
            subMatrix = np.rot90(subMatrix, k=1)

            # assign sub-matrix to appropriate cells in the result matrix
            assignX = i * side
            assignY = j * side
            result[assignX: assignX + side,
                   assignY: assignY + side] = subMatrix

    # assign axes and print the figure
    psm = axs.pcolormesh(
        result, cmap=cmVal, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(psm, ax=axs)
    plt.show()


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


def hypothesisLogisticRegression(theta, X):
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
    theta = theta.reshape(len(theta), 1)
    temp = np.dot(X, theta)
    return sigmoid(temp)


def costLogisticRegressionWithRegularization(theta, X, y, lambd):
    """
    Cost function for logistic regression with multiple features using
    regularization with parameter lambda.
    It calculates also gradient.

    Equation:
    cost = (1 / m) * sum((-Y * log(h))-((1 - Y)* log(1 - h)))
        + (lambda / 2m) * sum(theta .^ 2)
    grad = (1 / m) * (X' * (h - Y)) + (lambda / m) * theta

    m - size of training set
    n - number of features (including feature zero - 'bias')

    theta - the coefficent for the features (size n x 1)
    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    lambd - regularization parameter lambda

    return - a tuple with cost for given theta using regularization and gradient
    """
    m = len(y)
    h_x = hypothesisLogisticRegression(theta, X)
    theta = theta.reshape(len(theta), 1)
    leftTrue = -y * np.log(h_x)
    rightFalse = -(1-y) * np.log(1 - h_x)
    # regularized part does not include theta[0]
    regularizedPart = 0.5 * (lambd / m) * np.power(theta[1:], 2).sum()
    cost = (1.0 / m) * np.add(leftTrue, rightFalse).sum() + regularizedPart

    grad = np.zeros(np.shape(theta)[0])
    grad = grad.reshape(len(grad), 1)
    # for the first element of gradient we do not take into account regularization
    grad[0, 0] = (1.0 / m) * np.dot(np.transpose(X[:, 0]), h_x - y)
    # for all left elements, regularization part is calculated
    grad[1:] = (1.0 / m) * np.dot(np.transpose(X[:, 1:]),
                                  h_x - y) + (lambd / m) * theta[1:]
    grad = grad.flatten()
    return (cost, grad)


if __name__ == "__main__":
    # 20x20 Input Images of Digits
    input_layer_size = 400
    # 10 labels, from 1 to 10, (note that we have mapped "0" to label 10)
    num_labels = 10

    print('Loading and Visualizing Data ...\n')
    # it loads data from mathlab into dict with keys: 'X' and 'y', which
    # store data inside of np.ndarray
    data = scipy.io.loadmat('dataSet.mat')
    # size of data set - 5000
    m, n = data['X'].shape

    # create color map
    viridis = cm.get_cmap('viridis', 256)

    # randomly select 100 data points to display
    samples = 100
    random.seed()
    randIndices = random.sample(range(m), samples)
    tempSamples = np.ndarray(shape=(samples, n))
    for i, val in enumerate(randIndices):
        tempSamples[i] = data['X'][val, :]
    # plot the figure
    plotExamples(viridis, tempSamples)

    # test for cost and gradient
    theta_t = np.array(object=(-2, -1, 1, 2))
    temp = np.array([1.0, 1.0, 1.0, 1.0, 1.0] +
                    list(np.arange(0.1, 1.6, 0.1)), dtype=np.float64)
    X_t = temp.reshape(4, 5).transpose()
    y_t = np.array([[1], [0], [1], [0], [1]])
    lambda_t = 3
    (J, grad) = costLogisticRegressionWithRegularization(
        theta_t, X_t, y_t, lambda_t)

    print('\nCost: {0}'.format(J))
    print('Expected cost: 2.534819\n')
    print('Gradients:\n')
    print(' {0} '.format(grad))
    print('Expected gradients:')
    print(' 0.146561 -0.548558 0.724722 1.398003\n')
