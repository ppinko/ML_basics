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


def costLogisticRegressionWithRegularization(theta, X, Y, lambd):
    """
    Cost function for logistic regression with multiple features using
    regularization with parameter lambda.

    Equation:
    cost = (1 / m) * sum((-Y * log(h))-((1 - Y)* log(1 - h)))
        + (lambda / 2m) * sum(theta .^ 2)

    m - size of training set
    n - number of features (including feature zero - 'bias')

    theta - the coefficent for the features (size n x 1)
    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    lambd - regularization parameter lambda

    return - cost for given theta using regularization
    """
    m = len(Y)
    theta = theta.reshape(len(theta), 1)
    leftTrue = -Y * np.log(logReg.hypothesisLogisticRegression(theta, X))
    rightFalse = -(1-Y) * np.log(1 -
                                 logReg.hypothesisLogisticRegression(theta, X))
    regularizedPart = 0.5 * (lambd / m) * np.power(theta, 2).sum()
    return (1.0 / m) * np.add(leftTrue, rightFalse).sum() + regularizedPart


def gradientLogisticRegressionWithRegularization(theta, X, Y, lambd):
    """
    Calculates gradient for logistic regression with multiple features using
    regularization with parameter lambda.

    Equation:
    grad = (1 / m) * (X' * (h - Y)) + (lambda / m) * theta

    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)
    lambd - regularization parameter lambda

    return - optimal gradient
    """
    m = len(Y)
    n = len(theta)
    zeroFirst = np.ones(n)
    zeroFirst = theta.reshape(len(zeroFirst), 1)
    zeroFirst[0][0] = 0
    print(zeroFirst)
    theta = theta.reshape(len(theta), 1)

    grad = (1.0 / m) * np.dot(np.transpose(X),
                              logReg.hypothesisLogisticRegression(theta, X) - Y) + (lambd / m) * np.multiply(theta, zeroFirst)
    return grad.flatten()


def optimizeTheta(theta, X, Y, costFunction, gradientFunction, method, lambd):
    """
    Calculates optimal theta.

    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the initial coefficent for the features (size n x 1)
    costFunction - cost function
    gradientFunction - gradient function
    method - type of solver
    lambd - regularization parameter lambda


    return - optimal theta
    """
    result = op.minimize(fun=costFunction,
                         x0=theta,
                         args=(X, Y),
                         method=method,
                         jac=gradientFunction)
    return result.x


if __name__ == "__main__":
    # visualize data
    visualizeDataInitial1()

    # load data for testing
    df = pd.read_csv('microchipsQuailityAssurance.txt')
    X = np.array(df[['testA', 'testB']])
    X = np.column_stack((np.ones((len(X))), X))
    Y = np.array(df['decision'])
    Y = Y.reshape(len(Y), 1)

    # map features to 6th power
    mapX = mapFeature(X, degree=6)

    # test for cost function and gradient with all theta values set to 0
    # and lamba to 1
    thetaZero = np.zeros(np.shape(mapX)[1])
    initLambda = 1
    costForThetaZero = costLogisticRegressionWithRegularization(
        thetaZero, mapX, Y, initLambda)
    print('\nCost at initial theta(zeros): {0}'.format(costForThetaZero))
    print('Expected cost (approx): 0.693\n')
    gradForThetaZero = gradientLogisticRegressionWithRegularization(
        thetaZero, mapX, Y, initLambda)
    print('\nGradient at initial theta (zeros) - first five values only:')
    print(gradForThetaZero[:5])
    print('Expected gradients (approx) - first five values only:')
    print(' 0.0085 0.0188 0.0001 0.0503 0.0115\n')

    # test for cost function and gradient with all theta values set to 1
    # and lamba to 10
    thetaOne = np.ones(np.shape(mapX)[1])
    initLambda = 10
    costForThetaOne = costLogisticRegressionWithRegularization(
        thetaOne, mapX, Y, initLambda)
    print('\nCost at initial theta(ones): {0}'.format(costForThetaOne))
    print('Expected cost (approx): 3.16\n')
    gradForThetaOne = gradientLogisticRegressionWithRegularization(
        thetaOne, mapX, Y, initLambda)
    print('\nGradient at initial theta (ones) - first five values only:')
    print(gradForThetaOne[:5])
    print('Expected gradients (approx) - first five values only:')
    print(' 0.3460 0.1614 0.1948 0.2269 0.0922\n')
