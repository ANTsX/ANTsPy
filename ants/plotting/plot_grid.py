"""
Functions for plotting ants images
"""


__all__ = [
    "plot_grid"
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


def plot_grid(
    images,
    slices=None,
    axes=2,
    # general figure arguments
    figsize=1.0,
    rpad=0,
    cpad=0,
    vmin=None,
    vmax=None,
    colorbar=True,
    cmap="Greys_r",
    # title arguments
    title=None,
    tfontsize=20,
    title_dx=0,
    title_dy=0,
    # row arguments
    rlabels=None,
    rfontsize=14,
    rfontcolor="white",
    rfacecolor="black",
    # column arguments
    clabels=None,
    cfontsize=14,
    cfontcolor="white",
    cfacecolor="black",
    # save arguments
    filename=None,
    dpi=400,
    transparent=True,
    # other args
    **kwargs
):
    """
    Plot a collection of images in an arbitrarily-defined grid

    Matplotlib named colors: https://matplotlib.org/examples/color/named_colors.html

    Arguments
    ---------
    images : list of ANTsImage types
        image(s) to plot.
        if one image, this image will be used for all grid locations.
        if multiple images, they should be arrange in a list the same
        shape as the `gridsize` argument.

    slices : integer or list of integers
        slice indices to plot
        if one integer, this slice index will be used for all images
        if multiple integers, they should be arranged in a list the same
        shape as the `gridsize` argument

    axes : integer or list of integers
        axis or axes along which to plot image slices
        if one integer, this axis will be used for all images
        if multiple integers, they should be arranged in a list the same
        shape as the `gridsize` argument

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> mni1 = ants.image_read(ants.get_data('mni'))
    >>> mni2 = mni1.smooth_image(1.)
    >>> mni3 = mni1.smooth_image(2.)
    >>> mni4 = mni1.smooth_image(3.)
    >>> images = np.asarray([[mni1, mni2],
    ...                      [mni3, mni4]])
    >>> slices = np.asarray([[100, 100],
    ...                      [100, 100]])
    >>> ants.plot_grid(images=images, slices=slices, title='2x2 Grid')
    >>> images2d = np.asarray([[mni1.slice_image(2,100), mni2.slice_image(2,100)],
    ...                        [mni3.slice_image(2,100), mni4.slice_image(2,100)]])
    >>> ants.plot_grid(images=images2d, title='2x2 Grid Pre-Sliced')
    >>> ants.plot_grid(images.reshape(1,4), slices.reshape(1,4), title='1x4 Grid')
    >>> ants.plot_grid(images.reshape(4,1), slices.reshape(4,1), title='4x1 Grid')

    >>> # Padding between rows and/or columns
    >>> ants.plot_grid(images, slices, cpad=0.02, title='Col Padding')
    >>> ants.plot_grid(images, slices, rpad=0.02, title='Row Padding')
    >>> ants.plot_grid(images, slices, rpad=0.02, cpad=0.02, title='Row and Col Padding')

    >>> # Adding plain row and/or column labels
    >>> ants.plot_grid(images, slices, title='Adding Row Labels', rlabels=['Row #1', 'Row #2'])
    >>> ants.plot_grid(images, slices, title='Adding Col Labels', clabels=['Col #1', 'Col #2'])
    >>> ants.plot_grid(images, slices, title='Row and Col Labels',
                       rlabels=['Row 1', 'Row 2'], clabels=['Col 1', 'Col 2'])

    >>> # Making a publication-quality image
    >>> images = np.asarray([[mni1, mni2, mni2],
    ...                      [mni3, mni4, mni4]])
    >>> slices = np.asarray([[100, 100, 100],
    ...                      [100, 100, 100]])
    >>> axes = np.asarray([[0, 1, 2],
                           [0, 1, 2]])
    >>> ants.plot_grid(images, slices, axes, title='Publication Figures with ANTsPy',
                       tfontsize=20, title_dy=0.03, title_dx=-0.04,
                       rlabels=['Row 1', 'Row 2'],
                       clabels=['Col 1', 'Col 2', 'Col 3'],
                       rfontsize=16, cfontsize=16)
    """

    def mirror_matrix(x):
        return x[::-1, :]

    def rotate270_matrix(x):
        return mirror_matrix(x.T)

    def rotate180_matrix(x):
        return x[::-1, ::-1]

    def rotate90_matrix(x):
        return mirror_matrix(x).T

    def flip_matrix(x):
        return mirror_matrix(rotate180_matrix(x))

    def reorient_slice(x, axis):
        if axis != 1:
            x = rotate90_matrix(x)
        if axis == 1:
            x = rotate90_matrix(x)
        x = mirror_matrix(x)
        return x

    def slice_image(img, axis, idx):
        if axis == 0:
            return img[idx, :, :].numpy()
        elif axis == 1:
            return img[:, idx, :].numpy()
        elif axis == 2:
            return img[:, :, idx].numpy()
        elif axis == -1:
            return img[:, :, idx].numpy()
        elif axis == -2:
            return img[:, idx, :].numpy()
        elif axis == -3:
            return img[idx, :, :].numpy()
        else:
            raise ValueError("axis %i not valid" % axis)

    if isinstance(images, np.ndarray):
        images = images.tolist()
    if not isinstance(images, list):
        raise ValueError("images argument must be of type list")
    if not isinstance(images[0], list):
        images = [images]

    if slices is None:
        one_slice = True
    if isinstance(slices, int):
        one_slice = True
    if isinstance(slices, np.ndarray):
        slices = slices.tolist()
    if isinstance(slices, list):
        one_slice = False
        if not isinstance(slices[0], list):
            slices = [slices]
        nslicerow = len(slices)
        nslicecol = len(slices[0])

    nrow = len(images)
    ncol = len(images[0])

    if rlabels is None:
        rlabels = [None] * nrow
    if clabels is None:
        clabels = [None] * ncol

    if not one_slice:
        if (nrow != nslicerow) or (ncol != nslicecol):
            raise ValueError(
                "`images` arg shape (%i,%i) must equal `slices` arg shape (%i,%i)!"
                % (nrow, ncol, nslicerow, nslicecol)
            )

    fig = plt.figure(figsize=((ncol + 1) * 2.5 * figsize, (nrow + 1) * 2.5 * figsize))

    if title is not None:
        basex = 0.5
        basey = 0.9 if clabels[0] is None else 0.95
        fig.suptitle(title, fontsize=tfontsize, x=basex + title_dx, y=basey + title_dy)

    if (cpad > 0) and (rpad > 0):
        bothgridpad = max(cpad, rpad)
        cpad = 0
        rpad = 0
    else:
        bothgridpad = 0.0

    gs = gridspec.GridSpec(
        nrow,
        ncol,
        wspace=bothgridpad,
        hspace=0.0,
        top=1.0 - 0.5 / (nrow + 1),
        bottom=0.5 / (nrow + 1) + cpad,
        left=0.5 / (ncol + 1) + rpad,
        right=1 - 0.5 / (ncol + 1),
    )

    if isinstance(vmin, (int, float)):
        vmins = [vmin] * nrow
    elif vmin is None:
        vmins = [None] * nrow
    else:
        vmins = vmin

    if isinstance(vmax, (int, float)):
        vmaxs = [vmax] * nrow
    elif vmax is None:
        vmaxs = [None] * nrow
    else:
        vmaxs = vmax

    if isinstance(cmap, str):
        cmaps = [cmap] * nrow
    elif cmap is None:
        cmaps = [None] * nrow
    else:
        cmaps = cmap

    for rowidx, rvmin, rvmax, rcmap in zip(range(nrow), vmins, vmaxs, cmaps):
        for colidx in range(ncol):
            ax = plt.subplot(gs[rowidx, colidx])

            if colidx == 0:
                if rlabels[rowidx] is not None:
                    bottom, height = 0.25, 0.5
                    top = bottom + height
                    # add label text
                    ax.text(
                        -0.07,
                        0.5 * (bottom + top),
                        rlabels[rowidx],
                        horizontalalignment="right",
                        verticalalignment="center",
                        rotation="vertical",
                        transform=ax.transAxes,
                        color=rfontcolor,
                        fontsize=rfontsize,
                    )

                    # add label background
                    extra = 0.3 if rowidx == 0 else 0.0

                    rect = patches.Rectangle(
                        (-0.3, 0),
                        0.3,
                        1.0 + extra,
                        facecolor=rfacecolor,
                        alpha=1.0,
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                    ax.add_patch(rect)

            if rowidx == 0:
                if clabels[colidx] is not None:
                    bottom, height = 0.25, 0.5
                    left, width = 0.25, 0.5
                    right = left + width
                    top = bottom + height
                    ax.text(
                        0.5 * (left + right),
                        0.09 + top + bottom,
                        clabels[colidx],
                        horizontalalignment="center",
                        verticalalignment="center",
                        rotation="horizontal",
                        transform=ax.transAxes,
                        color=cfontcolor,
                        fontsize=cfontsize,
                    )

                    # add label background
                    rect = patches.Rectangle(
                        (0, 1.0),
                        1.0,
                        0.3,
                        facecolor=cfacecolor,
                        alpha=1.0,
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                    ax.add_patch(rect)

            tmpimg = images[rowidx][colidx]
            if isinstance(axes, int):
                tmpaxis = axes
            else:
                tmpaxis = axes[rowidx][colidx]
            
            if tmpimg.dimension == 2:
                tmpslice = tmpimg.numpy()
                tmpslice = reorient_slice(tmpslice, tmpaxis)
            else:
                sliceidx = slices[rowidx][colidx] if not one_slice else slices
                if sliceidx is None:
                    sliceidx = math.ceil(tmpimg.shape[tmpaxis] / 2)
                tmpslice = slice_image(tmpimg, tmpaxis, sliceidx)
                tmpslice = reorient_slice(tmpslice, tmpaxis)
            
            im = ax.imshow(tmpslice, cmap=rcmap, aspect="auto", vmin=rvmin, vmax=rvmax)
            ax.axis("off")

        # A colorbar solution with make_axes_locatable will not allow y-scaling of the colorbar.
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        if colorbar:
            axins = inset_axes(ax,
                               width="5%",  # width = 5% of parent_bbox width
                               height="90%",  # height : 50%
                               loc='center left',
                               bbox_to_anchor=(1.03, 0., 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0,
            )
            fig.colorbar(im, cax=axins, orientation='vertical')

    if filename is not None:
        filename = os.path.expanduser(filename)
        plt.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
