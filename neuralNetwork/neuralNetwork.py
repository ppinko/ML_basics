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
