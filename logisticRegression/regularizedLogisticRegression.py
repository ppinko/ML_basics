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


if __name__ == "__main__":
    # visualize data
    visualizeDataInitial1()
