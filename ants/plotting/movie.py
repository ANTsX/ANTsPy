"""
Functions for plotting ants images
"""


__all__ = [
    "movie"
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
from ants.decorators import image_method

@image_method
def movie(image, filename=None, writer=None, fps=30):
    """
    Create and save a movie - mp4, gif, etc - of the various
    2D slices of a 3D ants image

    Try this:
        conda install -c conda-forge ffmpeg

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> ants.movie(mni, filename='~/desktop/movie.mp4')
    """

    image = image.pad_image()
    img_arr = image.numpy()

    minidx = max(0, np.where(image > 0)[0][0] - 5)
    maxidx = max(image.shape[0], np.where(image > 0)[0][-1] + 5)

    # Creare your figure and axes
    fig, ax = plt.subplots(1)

    im = ax.imshow(
        img_arr[minidx, :, :],
        animated=True,
        cmap="Greys_r",
        vmin=image.quantile(0.05),
        vmax=image.quantile(0.95),
    )

    ax.axis("off")

    def init():
        fig.axes("off")
        return (im,)

    def updatefig(frame):
        im.set_array(img_arr[frame, :, :])
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=np.arange(minidx, maxidx),
        # init_func=init,
        interval=50,
        blit=True,
    )

    if writer is None:
        writer = animation.FFMpegWriter(fps=fps)

    if filename is not None:
        filename = os.path.expanduser(filename)
        ani.save(filename, writer=writer)
    else:
        plt.show()
