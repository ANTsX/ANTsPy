"""
Functions for plotting ants images
"""


__all__ = [
    "plot_hist"
]

import fnmatch
import math
import os
import warnings

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import numpy as np

from .. import registration as reg
from ..core import ants_image as iio
from ..core import ants_image_io as iio2
from ..core import ants_transform as tio
from ..core import ants_transform_io as tio2

def plot_hist(
    image,
    threshold=0.0,
    fit_line=False,
    normfreq=True,
    ## plot label arguments
    title=None,
    grid=True,
    xlabel=None,
    ylabel=None,
    ## other plot arguments
    facecolor="green",
    alpha=0.75,
):
    """
    Plot a histogram from an ANTsImage

    Arguments
    ---------
    image : ANTsImage
        image from which histogram will be created
    """
    img_arr = image.numpy().flatten()
    img_arr = img_arr[np.abs(img_arr) > threshold]

    if normfreq != False:
        normfreq = 1.0 if normfreq == True else normfreq
    n, bins, patches = plt.hist(
        img_arr, 50, facecolor=facecolor, alpha=alpha
    )

    if fit_line:
        # add a 'best fit' line
        y = mlab.normpdf(bins, img_arr.mean(), img_arr.std())
        l = plt.plot(bins, y, "r--", linewidth=1)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    plt.grid(grid)
    plt.show()
