import numpy as np
import pandas as pd


def costFunction(Y, X, theta):
    """
    Cost function for linear regression with multiple features.

    Equation:
    theta = (1/m) * sum((Y - X * theta)^2)
    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)

    return - cost for given theta and bias
    """
    size_training_set = len(Y)
    totalError = (np.power(np.subtract(Y, np.dot(X, theta)), 2)).sum()
    return 0.5 * totalError / size_training_set


def gradientDescent(Y, X, theta, alpha, num_iters):
    """
    Calculates theta (coefficients for features) using gradient descent
    algorithm.

    Equation:
    theta = theta - (alpha / m) * X' * (X * theta - Y)
    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)

    return - optimal theta (coefficients for features)
    """
    m = len(Y)
    for i in range(num_iters):
        theta = theta - (alpha / m) * \
            np.dot(np.transpose(X), (np.dot(X, theta) - Y))
    return theta


def predictOutcome(X, theta):
    """
    Predict outcome of the function using calculated theta (coefficients for
    features) for the given input data.

    Equation:
    theta = X * theta
    n - number of features (including feature zero - 'bias')

    X - the independent variables (features) (size 1 x n)
    theta - the coefficent for the features (size n x 1)

    return - prediction for optimal theta and given input
    """
    return np.dot(X, theta)


def normalEquationLinearRegression(Y, X):
    """
    Normal equation for linear regression for calculating optimat theta
    (coefficients for features).
    Equation:
    theta = (X' * X)^(-1) * X' * Y
    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)

    return - optimal theta (coefficients for features)
    """
    # temp1 = inv(X' * X)
    temp1 = np.linalg.inv(np.dot(np.transpose(X), X))
    # temp2 = X' * Y
    temp2 = np.dot(np.transpose(X), Y)
    return np.dot(temp1, temp2)


if __name__ == "__main__":
    df = pd.read_csv('data1.txt')
    X = np.array(df['X'])
    X = X.reshape(len(X), 1)
    X = np.column_stack((np.ones((len(X))), X))
    Y = np.array(df['Y'])
    Y = Y.reshape(len(Y), 1)

    # calculating cost
    thetaZero = np.zeros((2, 1))
    print(X.shape)
    costZero = costFunction(Y, X, thetaZero)
    print('With theta=[0, 0]')
    print('Cost function for theta[0, 0] = {0}'.format(costZero))
    print('Expected cost value (approx) 32.07\n')

    theta2 = np.array([[-1], [2]])
    cost2 = costFunction(Y, X, theta2)
    print('With theta=[-1, 2]')
    print('Cost function for theta[-1, 2] = {0}'.format(cost2))
    print('Expected cost value (approx) 54.24\n')

    # running gradient descent algorithm for finding optimal theta
    iterations = 1500
    alpha = 0.01
    gradient = gradientDescent(Y, X, thetaZero, alpha, iterations)
    print('Theta found by gradient descent:')
    print(gradient)
    print('Expected theta values (approx): -3.6303 1.1664\n')

    # calculating theta using normal equation
    thetaNormal = normalEquationLinearRegression(Y, X)
    print('Theta found by normal equation descent:')
    print(thetaNormal)

    # Linear regression with multiple variables
    df2 = pd.read_csv('data2.txt')
    X2 = np.array(df2[['X1', 'X2']])
    X2 = np.column_stack((np.ones((len(X2))), X2))
    Y2 = np.array(df2['Y'])
    Y2 = Y2.reshape(len(Y2), 1)

    # calculating theta using normal equation for multiple variables
    thetaNormal2 = normalEquationLinearRegression(Y2, X2)
    print('Theta found by normal equation descent:')
    print(thetaNormal2)

    # predict price of a 1650 sq-ft, 3 bedroom house
    toPredict = np.array([[1, 1650, 3]])
    predictedPrice = predictOutcome(toPredict, thetaNormal2)
    print(
        'Predicted price of a 1650 sq-ft, 3 br house (using normal equations): {0}'.format(predictedPrice))
