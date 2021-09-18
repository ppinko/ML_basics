import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

"""
viridis = cm.get_cmap('viridis', 12)
# print(viridis)
# print(viridis(0.56))
print('viridis.colors', viridis.colors)
# print('viridis(range(12))', viridis(range(12)))
# print('viridis(np.linspace(0, 1, 12))', viridis(np.linspace(0, 1, 12)))
"""

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([248/256, 24/256, 148/256, 1])
newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)


def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed()
    data = np.random.randn(30, 30)
    print(np.shape(data))

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()


plot_examples([viridis, newcmp])
