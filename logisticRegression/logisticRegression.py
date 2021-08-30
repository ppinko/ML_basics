import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op


def visualizeData1():
    """
    Visualize data for students admission to the university based on the results
    from two exams.
    """
    # load the dataset
    data = np.loadtxt('universityAdmission.txt', delimiter=',', skiprows=1)

    X = data[:, 0:2]
    y = data[:, 2]

    positive = np.where(y == 1)
    negative = np.where(y == 0)
    plt.scatter(X[positive, 0], X[positive, 1], marker='o', c='b')
    plt.scatter(X[negative, 0], X[negative, 1], marker='x', c='r')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Not Admitted', 'Admitted'])
    plt.title(label='University admission')
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


def costLogisticRegression(theta, X, Y):
    """
    Cost function for logistic regression with multiple features.

    Equation:
    cost = (1 / m) * sum((-Y * log(h))-((1 - Y)* log(1 - h)))

    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)

    return - cost for given theta
    """
    m = len(Y)
    theta = theta.reshape(len(theta), 1)
    leftTrue = -Y * np.log(hypothesisLogisticRegression(theta, X))
    rightFalse = -(1-Y) * np.log(1 - hypothesisLogisticRegression(theta, X))
    return (1.0 / m) * np.add(leftTrue, rightFalse).sum()


def gradientLogisticRegression(theta, X, Y):
    """
    Calculates gradient.

    Equation:
    grad = (1 / m) * (X' * (h - Y))
    m - size of training set
    n - number of features (including feature zero - 'bias')

    Y - output, target variable (size m x 1)
    X - the independent variables (features) (size m x n)
    theta - the coefficent for the features (size n x 1)

    return - optimal gradient
    """
    m = len(Y)
    theta = theta.reshape(len(theta), 1)
    grad = (1.0 / m) * np.dot(np.transpose(X),
                              hypothesisLogisticRegression(theta, X) - Y)
    return grad.flatten()


def optimizeTheta(theta, X, Y, costFunction, gradientFunction, method):
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

    return - optimal theta
    """
    result = op.minimize(fun=costFunction,
                         x0=theta,
                         args=(X, Y),
                         method=method,
                         jac=gradientFunction)
    return result.x


def predictOutcome(theta, X):
    """
    Predict outcome for a given data set and theta.

    m - size of training set
    n - number of features (including feature zero - 'bias')

    theta - the coefficent for the features (size n x 1)
    X - the independent variables (features) (size m x n)

    return - prediction for a given set of features
    """
    hypo = hypothesisLogisticRegression(theta, X)
    for i in range(len(hypo)):
        if hypo[i][0] >= 0.5:
            hypo[i][0] = 1
        else:
            hypo[i][0] = 0
    return hypo


def checkAccuracy(predictions, Y):
    """
    Check accuracy of the predictions.

    m - size of training set
    n - number of features (including feature zero - 'bias')

    predictions - predicted output (size m x 1)
    Y - output, target variable (size m x 1)

    return - accuracy of the predictions
    """
    m = len(Y)
    total = 0
    for i in range(m):
        if Y[i][0] == predictions[i][0]:
            total += 1
    return 100.0 * total / m


if __name__ == "__main__":
    # visualize data
    visualizeData1()

    # test sigmoid funciton
    arr = np.array([[-1, 0, 1]])
    sig = sigmoid(arr)
    print('Solution for sigmoid function for arr = [-1, 0, 1]:')
    print(' [{0}, {1}, {2}]\n'.format(sig[0][0], sig[0][1], sig[0][2]))

    # load data for testing
    df = pd.read_csv('universityAdmission.txt')
    X = np.array(df[['scoreA', 'scoreB']])
    X = np.column_stack((np.ones((len(X))), X))
    Y = np.array(df['decision'])
    Y = Y.reshape(len(Y), 1)

    # test for cost function and gradient
    thetaZero = np.zeros(3)
    initialCost = costLogisticRegression(thetaZero, X, Y)
    initialGradient = gradientLogisticRegression(thetaZero, X, Y)
    print('Cost at initial theta (zeros): {0}'.format(initialCost))
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros):')
    print('Result = {0} {1} {2}'.format(
        initialGradient[0], initialGradient[1], initialGradient[2]))
    print('Expected gradients (approx): -0.1000 -12.0092 -11.2628\n')

    # use scipy optimization for finding optimal theta
    optimizedTheta = optimizeTheta(
        thetaZero, X, Y, costLogisticRegression, gradientLogisticRegression, 'TNC')
    print('Calculation using optimization methods from scipy:')
    print('Result = {0} {1} {2}'.format(
        optimizedTheta[0], optimizedTheta[1], optimizedTheta[2]))
    print('Expected gradients (approx): -25.161 0.206 0.201\n')

    # predict probability for a student with scores 45 and 85
    singleX = np.array([[1.0, 45.0, 85.0]])
    probability = hypothesisLogisticRegression(optimizedTheta, singleX)
    print('For a student with scores 45 and 85, we predict an admission probability of {0}'.format(
        probability[0][0]))
    print('Expected value: 0.775 +/- 0.002\n')

    # predict accuracy using calculated theta and given data set
    predictions = predictOutcome(optimizedTheta, X)
    accuracy = checkAccuracy(predictions, Y)
    print('Train Accuracy: {0}'.format(accuracy))
    print('Expected accuracy (approx): 89.0\n')
