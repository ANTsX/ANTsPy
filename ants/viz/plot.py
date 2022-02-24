"""
Create a static 2D image of a 2D ANTsImage
or a tile of slices from a 3D ANTsImage

TODO:
- add `plot_multichannel` function for plotting multi-channel images
    - support for quivers as well
- add `plot_gif` function for making a gif/video or 2D slices across a 3D image
"""


__all__ = [
    "plot",
    "movie",
    "plot_hist",
    "plot_grid",
    "plot_ortho",
    "plot_ortho_double",
    "plot_ortho_stack",
    "plot_directory",
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
    >>> #axes = np.asarray([[2,2],[2,2]])
    >>> # standard plotting
    >>> ants.plot_grid(images=images, slices=slices, title='2x2 Grid')
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
            return img[idx, :, :]
        elif axis == 1:
            return img[:, idx, :]
        elif axis == 2:
            return img[:, :, idx]
        elif axis == -1:
            return img[:, :, idx]
        elif axis == -2:
            return img[:, idx, :]
        elif axis == -3:
            return img[idx, :, :]
        else:
            raise ValueError("axis %i not valid" % axis)

    if isinstance(images, np.ndarray):
        images = images.tolist()
    if not isinstance(images, list):
        raise ValueError("images argument must be of type list")
    if not isinstance(images[0], list):
        images = [images]

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
            sliceidx = slices[rowidx][colidx] if not one_slice else slices
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


def plot_ortho_stack(
    images,
    overlays=None,
    reorient=True,
    # xyz arguments
    xyz=None,
    xyz_lines=False,
    xyz_color="red",
    xyz_alpha=0.6,
    xyz_linewidth=2,
    xyz_pad=5,
    # base image arguments
    cmap="Greys_r",
    alpha=1,
    # overlay arguments
    overlay_cmap="jet",
    overlay_alpha=0.9,
    # background arguments
    black_bg=True,
    bg_thresh_quant=0.01,
    bg_val_quant=0.99,
    # scale/crop/domain arguments
    crop=False,
    scale=False,
    domain_image_map=None,
    # title arguments
    title=None,
    titlefontsize=24,
    title_dx=0,
    title_dy=0,
    # 4th panel text arguemnts
    text=None,
    textfontsize=24,
    textfontcolor="white",
    text_dx=0,
    text_dy=0,
    # save & size arguments
    filename=None,
    dpi=500,
    figsize=1.0,
    colpad=0,
    rowpad=0,
    transpose=False,
    transparent=True,
    orient_labels=True,
):
    """
    Create a stack of orthographic plots with optional overlays.

    Use mask_image and/or threshold_image to preprocess images to be be
    overlaid and display the overlays in a given range. See the wiki examples.

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> ch2 = ants.image_read(ants.get_data('ch2'))
    >>> ants.plot_ortho_stack([mni,mni,mni])
    """

    def mirror_matrix(x):
        return x[::-1, :]

    def rotate270_matrix(x):
        return mirror_matrix(x.T)

    def reorient_slice(x, axis):
        return rotate270_matrix(x)

    # need this hack because of a weird NaN warning from matplotlib with overlays
    warnings.simplefilter("ignore")

    n_images = len(images)

    # handle `image` argument
    for i in range(n_images):
        if isinstance(images[i], str):
            images[i] = iio2.image_read(images[i])
        if not isinstance(images[i], iio.ANTsImage):
            raise ValueError("image argument must be an ANTsImage")
        if images[i].dimension != 3:
            raise ValueError("Input image must have 3 dimensions!")

    if overlays is None:
        overlays = [None] * n_images
    # handle `overlay` argument
    for i in range(n_images):
        if overlays[i] is not None:
            if isinstance(overlays[i], str):
                overlays[i] = iio2.image_read(overlays[i])
            if not isinstance(overlays[i], iio.ANTsImage):
                raise ValueError("overlay argument must be an ANTsImage")
            if overlays[i].dimension != 3:
                raise ValueError("Overlay image must have 3 dimensions!")

            if not iio.image_physical_space_consistency(images[i], overlays[i]):
                overlays[i] = reg.resample_image_to_target(
                    overlays[i], images[i], interp_type="linear"
                )

    for i in range(1, n_images):
        if not iio.image_physical_space_consistency(images[0], images[i]):
            images[i] = reg.resample_image_to_target(
                images[0], images[i], interp_type="linear"
            )

    # reorient images
    if reorient != False:
        if reorient == True:
            reorient = "RPI"

        for i in range(n_images):
            images[i] = images[i].reorient_image2(reorient)

        if overlays[i] is not None:
            overlays[i] = overlays[i].reorient_image2(reorient)

    # handle `slices` argument
    if xyz is None:
        xyz = [int(s / 2) for s in images[0].shape]
    for i in range(3):
        if xyz[i] is None:
            xyz[i] = int(images[0].shape[i] / 2)

    # resample image if spacing is very unbalanced
    spacing = [s for i, s in enumerate(images[0].spacing)]
    if (max(spacing) / min(spacing)) > 3.0:
        new_spacing = (1, 1, 1)
        for i in range(n_images):
            images[i] = images[i].resample_image(tuple(new_spacing))
        if overlays[i] is not None:
            overlays[i] = overlays[i].resample_image(tuple(new_spacing))
        xyz = [
            int(sl * (sold / snew)) for sl, sold, snew in zip(xyz, spacing, new_spacing)
        ]

    # potentially crop image
    if crop:
        for i in range(n_images):
            plotmask = images[i].get_mask(cleanup=0)
            if plotmask.max() == 0:
                plotmask += 1
            images[i] = images[i].crop_image(plotmask)
            if overlays[i] is not None:
                overlays[i] = overlays[i].crop_image(plotmask)

    # pad images
    for i in range(n_images):
        if i == 0:
            images[i], lowpad, uppad = images[i].pad_image(return_padvals=True)
        else:
            images[i] = images[i].pad_image()
        if overlays[i] is not None:
            overlays[i] = overlays[i].pad_image()
    xyz = [v + l for v, l in zip(xyz, lowpad)]

    # handle `domain_image_map` argument
    if domain_image_map is not None:
        if isinstance(domain_image_map, iio.ANTsImage):
            tx = tio2.new_ants_transform(
                precision="float", transform_type="AffineTransform", dimension=3
            )
            for i in range(n_images):
                images[i] = tio.apply_ants_transform_to_image(
                    tx, images[i], domain_image_map
                )

                if overlays[i] is not None:
                    overlays[i] = tio.apply_ants_transform_to_image(
                        tx, overlays[i], domain_image_map, interpolation="linear"
                    )
        elif isinstance(domain_image_map, (list, tuple)):
            # expect an image and transformation
            if len(domain_image_map) != 2:
                raise ValueError("domain_image_map list or tuple must have length == 2")

            dimg = domain_image_map[0]
            if not isinstance(dimg, iio.ANTsImage):
                raise ValueError("domain_image_map first entry should be ANTsImage")

            tx = domain_image_map[1]
            for i in range(n_images):
                images[i] = reg.apply_transforms(dimg, images[i], transform_list=tx)
                if overlays[i] is not None:
                    overlays[i] = reg.apply_transforms(
                        dimg, overlays[i], transform_list=tx, interpolator="linear"
                    )

    # potentially find dynamic range
    if scale == True:
        vmins = []
        vmaxs = []
        for i in range(n_images):
            vmin, vmax = images[i].quantile((0.05, 0.95))
            vmins.append(vmin)
            vmaxs.append(vmax)
    elif isinstance(scale, (list, tuple)):
        if len(scale) != 2:
            raise ValueError(
                "scale argument must be boolean or list/tuple with two values"
            )
        vmins = []
        vmaxs = []
        for i in range(n_images):
            vmin, vmax = images[i].quantile(scale)
            vmins.append(vmin)
            vmaxs.append(vmax)
    else:
        vmin = None
        vmax = None

    if not transpose:
        nrow = n_images
        ncol = 3
    else:
        nrow = 3
        ncol = n_images

    fig = plt.figure(figsize=((ncol + 1) * 2.5 * figsize, (nrow + 1) * 2.5 * figsize))
    if title is not None:
        basey = 0.93
        basex = 0.5
        fig.suptitle(
            title, fontsize=titlefontsize, x=basex + title_dx, y=basey + title_dy
        )

    if (colpad > 0) and (rowpad > 0):
        bothgridpad = max(colpad, rowpad)
        colpad = 0
        rowpad = 0
    else:
        bothgridpad = 0.0

    gs = gridspec.GridSpec(
        nrow,
        ncol,
        wspace=bothgridpad,
        hspace=0.0,
        top=1.0 - 0.5 / (nrow + 1),
        bottom=0.5 / (nrow + 1) + colpad,
        left=0.5 / (ncol + 1) + rowpad,
        right=1 - 0.5 / (ncol + 1),
    )

    # pad image to have isotropic array dimensions
    vminols=[]
    vmaxols=[]
    for i in range(n_images):
        images[i] = images[i].numpy()
        if overlays[i] is not None:
            vminols.append( overlays[i].min() )
            vmaxols.append( overlays[i].max() )
            overlays[i] = overlays[i].numpy()
            if overlays[i].dtype not in ["uint8", "uint32"]:
                overlays[i][np.abs(overlays[i]) == 0] = np.nan

    ####################
    ####################
    for i in range(n_images):
        yz_slice = reorient_slice(images[i][xyz[0], :, :], 0)
        if not transpose:
            ax = plt.subplot(gs[i, 0])
        else:
            ax = plt.subplot(gs[0, i])
        ax.imshow(yz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlays[i] is not None:
            yz_overlay = reorient_slice(overlays[i][xyz[0], :, :], 0)
            ax.imshow(yz_overlay, alpha=overlay_alpha, cmap=overlay_cmap,
                vmin=vminols[i], vmax=vmaxols[i])
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [yz_slice.shape[0] - xyz[1], yz_slice.shape[0] - xyz[1]],
                [xyz_pad, yz_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, yz_slice.shape[1] - xyz_pad],
                [yz_slice.shape[1] - xyz[2], yz_slice.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        if orient_labels:
            ax.text(
                0.5,
                0.98,
                "S",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.02,
                "I",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.98,
                0.5,
                "A",
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.5,
                "P",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
        ax.axis("off")
        ####################
        ####################

        xz_slice = reorient_slice(images[i][:, xyz[1], :], 1)
        if not transpose:
            ax = plt.subplot(gs[i, 1])
        else:
            ax = plt.subplot(gs[1, i])
        ax.imshow(xz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlays[i] is not None:
            xz_overlay = reorient_slice(overlays[i][:, xyz[1], :], 1)
            ax.imshow(xz_overlay, alpha=overlay_alpha, cmap=overlay_cmap,
                vmin=vminols[i], vmax=vmaxols[i])
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xz_slice.shape[0] - xyz[0], xz_slice.shape[0] - xyz[0]],
                [xyz_pad, xz_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xz_slice.shape[1] - xyz_pad],
                [xz_slice.shape[1] - xyz[2], xz_slice.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        if orient_labels:
            ax.text(
                0.5,
                0.98,
                "A",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.02,
                "P",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.98,
                0.5,
                "L",
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.5,
                "R",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
        ax.axis("off")

        ####################
        ####################
        xy_slice = reorient_slice(images[i][:, :, xyz[2]], 2)
        if not transpose:
            ax = plt.subplot(gs[i, 2])
        else:
            ax = plt.subplot(gs[2, i])
        ax.imshow(xy_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlays[i] is not None:
            xy_overlay = reorient_slice(overlays[i][:, :, xyz[2]], 2)
            ax.imshow(xy_overlay, alpha=overlay_alpha, cmap=overlay_cmap,
                vmin=vminols[i], vmax=vmaxols[i])
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xy_slice.shape[0] - xyz[0], xy_slice.shape[0] - xyz[0]],
                [xyz_pad, xy_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xy_slice.shape[1] - xyz_pad],
                [xy_slice.shape[1] - xyz[1], xy_slice.shape[1] - xyz[1]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        if orient_labels:
            ax.text(
                0.5,
                0.98,
                "A",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.02,
                "P",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.98,
                0.5,
                "L",
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.5,
                "R",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
        ax.axis("off")

    ####################
    ####################

    if filename is not None:
        plt.savefig(filename, dpi=dpi, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter("default")


def plot_ortho_double(
    image,
    image2,
    overlay=None,
    overlay2=None,
    reorient=True,
    # xyz arguments
    xyz=None,
    xyz_lines=True,
    xyz_color="red",
    xyz_alpha=0.6,
    xyz_linewidth=2,
    xyz_pad=5,
    # base image arguments
    cmap="Greys_r",
    alpha=1,
    cmap2="Greys_r",
    alpha2=1,
    # overlay arguments
    overlay_cmap="jet",
    overlay_alpha=0.9,
    overlay_cmap2="jet",
    overlay_alpha2=0.9,
    # background arguments
    black_bg=True,
    bg_thresh_quant=0.01,
    bg_val_quant=0.99,
    # scale/crop/domain arguments
    crop=False,
    scale=False,
    crop2=False,
    scale2=True,
    domain_image_map=None,
    # title arguments
    title=None,
    titlefontsize=24,
    title_dx=0,
    title_dy=0,
    # 4th panel text arguemnts
    text=None,
    textfontsize=24,
    textfontcolor="white",
    text_dx=0,
    text_dy=0,
    # save & size arguments
    filename=None,
    dpi=500,
    figsize=1.0,
    flat=True,
    transpose=False,
    transparent=True,
):
    """
    Create a pair of orthographic plots with overlays.

    Use mask_image and/or threshold_image to preprocess images to be be
    overlaid and display the overlays in a given range. See the wiki examples.

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> ch2 = ants.image_read(ants.get_data('ch2'))
    >>> ants.plot_ortho_double(mni, ch2)
    """

    def mirror_matrix(x):
        return x[::-1, :]

    def rotate270_matrix(x):
        return mirror_matrix(x.T)

    def reorient_slice(x, axis):
        return rotate270_matrix(x)

    # need this hack because of a weird NaN warning from matplotlib with overlays
    warnings.simplefilter("ignore")

    # handle `image` argument
    if isinstance(image, str):
        image = iio2.image_read(image)
    if not isinstance(image, iio.ANTsImage):
        raise ValueError("image argument must be an ANTsImage")
    if image.dimension != 3:
        raise ValueError("Input image must have 3 dimensions!")

    if isinstance(image2, str):
        image2 = iio2.image_read(image2)
    if not isinstance(image2, iio.ANTsImage):
        raise ValueError("image2 argument must be an ANTsImage")
    if image2.dimension != 3:
        raise ValueError("Input image2 must have 3 dimensions!")

    # handle `overlay` argument
    if overlay is not None:
        if isinstance(overlay, str):
            overlay = iio2.image_read(overlay)
        if not isinstance(overlay, iio.ANTsImage):
            raise ValueError("overlay argument must be an ANTsImage")
        if overlay.dimension != 3:
            raise ValueError("Overlay image must have 3 dimensions!")

        if not iio.image_physical_space_consistency(image, overlay):
            overlay = reg.resample_image_to_target(overlay, image, interp_type="linear")

    if overlay2 is not None:
        if isinstance(overlay2, str):
            overlay2 = iio2.image_read(overlay2)
        if not isinstance(overlay2, iio.ANTsImage):
            raise ValueError("overlay2 argument must be an ANTsImage")
        if overlay2.dimension != 3:
            raise ValueError("Overlay2 image must have 3 dimensions!")

        if not iio.image_physical_space_consistency(image2, overlay2):
            overlay2 = reg.resample_image_to_target(
                overlay2, image2, interp_type="linear"
            )

    if not iio.image_physical_space_consistency(image, image2):
        image2 = reg.resample_image_to_target(image2, image, interp_type="linear")

    if image.pixeltype not in {"float", "double"}:
        scale = False  # turn off scaling if image is discrete

    if image2.pixeltype not in {"float", "double"}:
        scale2 = False  # turn off scaling if image is discrete

    # reorient images
    if reorient != False:
        if reorient == True:
            reorient = "RPI"
        image = image.reorient_image2(reorient)
        image2 = image2.reorient_image2(reorient)
        if overlay is not None:
            overlay = overlay.reorient_image2(reorient)
        if overlay2 is not None:
            overlay2 = overlay2.reorient_image2(reorient)

    # handle `slices` argument
    if xyz is None:
        xyz = [int(s / 2) for s in image.shape]
    for i in range(3):
        if xyz[i] is None:
            xyz[i] = int(image.shape[i] / 2)

    # resample image if spacing is very unbalanced
    spacing = [s for i, s in enumerate(image.spacing)]
    if (max(spacing) / min(spacing)) > 3.0:
        new_spacing = (1, 1, 1)
        image = image.resample_image(tuple(new_spacing))
        image2 = image2.resample_image_to_target(tuple(new_spacing))
        if overlay is not None:
            overlay = overlay.resample_image(tuple(new_spacing))
        if overlay2 is not None:
            overlay2 = overlay2.resample_image(tuple(new_spacing))
        xyz = [
            int(sl * (sold / snew)) for sl, sold, snew in zip(xyz, spacing, new_spacing)
        ]

    # pad images
    image, lowpad, uppad = image.pad_image(return_padvals=True)
    image2, lowpad2, uppad2 = image2.pad_image(return_padvals=True)
    xyz = [v + l for v, l in zip(xyz, lowpad)]
    if overlay is not None:
        overlay = overlay.pad_image()
    if overlay2 is not None:
        overlay2 = overlay2.pad_image()

    # handle `domain_image_map` argument
    if domain_image_map is not None:
        if isinstance(domain_image_map, iio.ANTsImage):
            tx = tio2.new_ants_transform(
                precision="float",
                transform_type="AffineTransform",
                dimension=image.dimension,
            )
            image = tio.apply_ants_transform_to_image(tx, image, domain_image_map)
            image2 = tio.apply_ants_transform_to_image(tx, image2, domain_image_map)
            if overlay is not None:
                overlay = tio.apply_ants_transform_to_image(
                    tx, overlay, domain_image_map, interpolation="linear"
                )
            if overlay2 is not None:
                overlay2 = tio.apply_ants_transform_to_image(
                    tx, overlay2, domain_image_map, interpolation="linear"
                )
        elif isinstance(domain_image_map, (list, tuple)):
            # expect an image and transformation
            if len(domain_image_map) != 2:
                raise ValueError("domain_image_map list or tuple must have length == 2")

            dimg = domain_image_map[0]
            if not isinstance(dimg, iio.ANTsImage):
                raise ValueError("domain_image_map first entry should be ANTsImage")

            tx = domain_image_map[1]
            image = reg.apply_transforms(dimg, image, transform_list=tx)
            if overlay is not None:
                overlay = reg.apply_transforms(
                    dimg, overlay, transform_list=tx, interpolator="linear"
                )

            image2 = reg.apply_transforms(dimg, image2, transform_list=tx)
            if overlay2 is not None:
                overlay2 = reg.apply_transforms(
                    dimg, overlay2, transform_list=tx, interpolator="linear"
                )

    ## single-channel images ##
    if image.components == 1:

        # potentially crop image
        if crop:
            plotmask = image.get_mask(cleanup=0)
            if plotmask.max() == 0:
                plotmask += 1
            image = image.crop_image(plotmask)
            if overlay is not None:
                overlay = overlay.crop_image(plotmask)

        if crop2:
            plotmask2 = image2.get_mask(cleanup=0)
            if plotmask2.max() == 0:
                plotmask2 += 1
            image2 = image2.crop_image(plotmask2)
            if overlay2 is not None:
                overlay2 = overlay2.crop_image(plotmask2)

        # potentially find dynamic range
        if scale == True:
            vmin, vmax = image.quantile((0.05, 0.95))
        elif isinstance(scale, (list, tuple)):
            if len(scale) != 2:
                raise ValueError(
                    "scale argument must be boolean or list/tuple with two values"
                )
            vmin, vmax = image.quantile(scale)
        else:
            vmin = None
            vmax = None

        if scale2 == True:
            vmin2, vmax2 = image2.quantile((0.05, 0.95))
        elif isinstance(scale2, (list, tuple)):
            if len(scale2) != 2:
                raise ValueError(
                    "scale2 argument must be boolean or list/tuple with two values"
                )
            vmin2, vmax2 = image2.quantile(scale2)
        else:
            vmin2 = None
            vmax2 = None

        if not flat:
            nrow = 2
            ncol = 4
        else:
            if not transpose:
                nrow = 2
                ncol = 3
            else:
                nrow = 3
                ncol = 2

        fig = plt.figure(
            figsize=((ncol + 1) * 2.5 * figsize, (nrow + 1) * 2.5 * figsize)
        )
        if title is not None:
            basey = 0.88 if not flat else 0.66
            basex = 0.5
            fig.suptitle(
                title, fontsize=titlefontsize, x=basex + title_dx, y=basey + title_dy
            )

        gs = gridspec.GridSpec(
            nrow,
            ncol,
            wspace=0.0,
            hspace=0.0,
            top=1.0 - 0.5 / (nrow + 1),
            bottom=0.5 / (nrow + 1),
            left=0.5 / (ncol + 1),
            right=1 - 0.5 / (ncol + 1),
        )

        # pad image to have isotropic array dimensions
        image = image.numpy()
        if overlay is not None:
            overlay = overlay.numpy()
            if overlay.dtype not in ["uint8", "uint32"]:
                overlay[np.abs(overlay) == 0] = np.nan

        image2 = image2.numpy()
        if overlay2 is not None:
            overlay2 = overlay2.numpy()
            if overlay2.dtype not in ["uint8", "uint32"]:
                overlay2[np.abs(overlay2) == 0] = np.nan

        ####################
        ####################
        yz_slice = reorient_slice(image[xyz[0], :, :], 0)
        ax = plt.subplot(gs[0, 0])
        ax.imshow(yz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            yz_overlay = reorient_slice(overlay[xyz[0], :, :], 0)
            ax.imshow(yz_overlay, alpha=overlay_alpha, cmap=overlay_cmap)
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [yz_slice.shape[0] - xyz[1], yz_slice.shape[0] - xyz[1]],
                [xyz_pad, yz_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, yz_slice.shape[1] - xyz_pad],
                [yz_slice.shape[1] - xyz[2], yz_slice.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        ax.axis("off")

        #######
        yz_slice2 = reorient_slice(image2[xyz[0], :, :], 0)
        if not flat:
            ax = plt.subplot(gs[0, 1])
        else:
            if not transpose:
                ax = plt.subplot(gs[1, 0])
            else:
                ax = plt.subplot(gs[0, 1])
        ax.imshow(yz_slice2, cmap=cmap2, vmin=vmin2, vmax=vmax2)
        if overlay2 is not None:
            yz_overlay2 = reorient_slice(overlay2[xyz[0], :, :], 0)
            ax.imshow(yz_overlay2, alpha=overlay_alpha2, cmap=overlay_cmap2)
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [yz_slice2.shape[0] - xyz[1], yz_slice2.shape[0] - xyz[1]],
                [xyz_pad, yz_slice2.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, yz_slice2.shape[1] - xyz_pad],
                [yz_slice2.shape[1] - xyz[2], yz_slice2.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        ax.axis("off")
        ####################
        ####################

        xz_slice = reorient_slice(image[:, xyz[1], :], 1)
        if not flat:
            ax = plt.subplot(gs[0, 2])
        else:
            if not transpose:
                ax = plt.subplot(gs[0, 1])
            else:
                ax = plt.subplot(gs[1, 0])
        ax.imshow(xz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            xz_overlay = reorient_slice(overlay[:, xyz[1], :], 1)
            ax.imshow(xz_overlay, alpha=overlay_alpha, cmap=overlay_cmap)
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xz_slice.shape[0] - xyz[0], xz_slice.shape[0] - xyz[0]],
                [xyz_pad, xz_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xz_slice.shape[1] - xyz_pad],
                [xz_slice.shape[1] - xyz[2], xz_slice.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        ax.axis("off")

        #######
        xz_slice2 = reorient_slice(image2[:, xyz[1], :], 1)
        if not flat:
            ax = plt.subplot(gs[0, 3])
        else:
            ax = plt.subplot(gs[1, 1])
        ax.imshow(xz_slice2, cmap=cmap2, vmin=vmin2, vmax=vmax2)
        if overlay is not None:
            xz_overlay2 = reorient_slice(overlay2[:, xyz[1], :], 1)
            ax.imshow(xz_overlay2, alpha=overlay_alpha2, cmap=overlay_cmap2)
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xz_slice2.shape[0] - xyz[0], xz_slice2.shape[0] - xyz[0]],
                [xyz_pad, xz_slice2.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xz_slice2.shape[1] - xyz_pad],
                [xz_slice2.shape[1] - xyz[2], xz_slice2.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        ax.axis("off")

        ####################
        ####################
        xy_slice = reorient_slice(image[:, :, xyz[2]], 2)
        if not flat:
            ax = plt.subplot(gs[1, 2])
        else:
            if not transpose:
                ax = plt.subplot(gs[0, 2])
            else:
                ax = plt.subplot(gs[2, 0])
        ax.imshow(xy_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            xy_overlay = reorient_slice(overlay[:, :, xyz[2]], 2)
            ax.imshow(xy_overlay, alpha=overlay_alpha, cmap=overlay_cmap)
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xy_slice.shape[0] - xyz[0], xy_slice.shape[0] - xyz[0]],
                [xyz_pad, xy_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xy_slice.shape[1] - xyz_pad],
                [xy_slice.shape[1] - xyz[1], xy_slice.shape[1] - xyz[1]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        ax.axis("off")

        #######
        xy_slice2 = reorient_slice(image2[:, :, xyz[2]], 2)
        if not flat:
            ax = plt.subplot(gs[1, 3])
        else:
            if not transpose:
                ax = plt.subplot(gs[1, 2])
            else:
                ax = plt.subplot(gs[2, 1])
        ax.imshow(xy_slice2, cmap=cmap2, vmin=vmin2, vmax=vmax2)
        if overlay is not None:
            xy_overlay2 = reorient_slice(overlay2[:, :, xyz[2]], 2)
            ax.imshow(xy_overlay2, alpha=overlay_alpha2, cmap=overlay_cmap2)
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xy_slice2.shape[0] - xyz[0], xy_slice2.shape[0] - xyz[0]],
                [xyz_pad, xy_slice2.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xy_slice2.shape[1] - xyz_pad],
                [xy_slice2.shape[1] - xyz[1], xy_slice2.shape[1] - xyz[1]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        ax.axis("off")

        ####################
        ####################

        if not flat:
            # empty corner
            ax = plt.subplot(gs[1, :2])
            if text is not None:
                # add text
                left, width = 0.25, 0.5
                bottom, height = 0.25, 0.5
                right = left + width
                top = bottom + height
                ax.text(
                    0.5 * (left + right) + text_dx,
                    0.5 * (bottom + top) + text_dy,
                    text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=textfontsize,
                    color=textfontcolor,
                    transform=ax.transAxes,
                )
            # ax.text(0.5, 0.5)
            img_shape = list(image.shape[:-1])
            img_shape[1] *= 2
            ax.imshow(np.zeros(img_shape), cmap="Greys_r")
            ax.axis("off")

    ## multi-channel images ##
    elif image.components > 1:
        raise ValueError("Multi-channel images not currently supported!")

    if filename is not None:
        plt.savefig(filename, dpi=dpi, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter("default")


def plot_ortho(
    image,
    overlay=None,
    reorient=True,
    blend=False,
    # xyz arguments
    xyz=None,
    xyz_lines=True,
    xyz_color="red",
    xyz_alpha=0.6,
    xyz_linewidth=2,
    xyz_pad=5,
    orient_labels=True,
    # base image arguments
    alpha=1,
    cmap="Greys_r",
    # overlay arguments
    overlay_cmap="jet",
    overlay_alpha=0.9,
    # background arguments
    black_bg=True,
    bg_thresh_quant=0.01,
    bg_val_quant=0.99,
    # scale/crop/domain arguments
    crop=False,
    scale=False,
    domain_image_map=None,
    # title arguments
    title=None,
    titlefontsize=24,
    title_dx=0,
    title_dy=0,
    # 4th panel text arguemnts
    text=None,
    textfontsize=24,
    textfontcolor="white",
    text_dx=0,
    text_dy=0,
    # save & size arguments
    filename=None,
    dpi=500,
    figsize=1.0,
    flat=False,
    transparent=True,
):
    """
    Plot an orthographic view of a 3D image

    Use mask_image and/or threshold_image to preprocess images to be be
    overlaid and display the overlays in a given range. See the wiki examples.

    TODO:
        - add colorbar option

    ANTsR function: N/A

    Arguments
    ---------
    image : ANTsImage
        image to plot

    overlay : ANTsImage
        image to overlay on base image

    slices : list or tuple of 3 integers
        slice indices along each axis to plot
        This can be absolute array indices (e.g. (80,100,120)), or
        this can be relative array indices (e.g. (0.4,0.5,0.6)).
        The default is to take the middle slice along each axis.

    xyz : list or tuple of 3 integers
        if given, solid lines will be drawn to converge at this coordinate.
        This is useful for pinpointing a specific location in the image.

    flat : boolean
        if true, the ortho image will be plot in one row
        if false, the ortho image will be a 2x2 grid with the bottom
            left corner blank

    cmap : string
        colormap to use for base image. See matplotlib.

    overlay_cmap : string
        colormap to use for overlay images, if applicable. See matplotlib.

    overlay_alpha : float
        level of transparency for any overlays. Smaller value means
        the overlay is more transparent. See matplotlib.

    axis : integer
        which axis to plot along if image is 3D

    black_bg : boolean
        if True, the background of the image(s) will be black.
        if False, the background of the image(s) will be determined by the
            values `bg_thresh_quant` and `bg_val_quant`.

    bg_thresh_quant : float
        if white_bg=True, the background will be determined by thresholding
        the image at the `bg_thresh` quantile value and setting the background
        intensity to the `bg_val` quantile value.
        This value should be in [0, 1] - somewhere around 0.01 is recommended.
            - equal to 1 will threshold the entire image
            - equal to 0 will threshold none of the image

    bg_val_quant : float
        if white_bg=True, the background will be determined by thresholding
        the image at the `bg_thresh` quantile value and setting the background
        intensity to the `bg_val` quantile value.
        This value should be in [0, 1]
            - equal to 1 is pure white
            - equal to 0 is pure black
            - somewhere in between is gray

    domain_image_map : ANTsImage
        this input ANTsImage or list of ANTsImage types contains a reference image
        `domain_image` and optional reference mapping named `domainMap`.
        If supplied, the image(s) to be plotted will be mapped to the domain
        image space before plotting - useful for non-standard image orientations.

    crop : boolean
        if true, the image(s) will be cropped to their bounding boxes, resulting
        in a potentially smaller image size.
        if false, the image(s) will not be cropped

    scale : boolean or 2-tuple
        if true, nothing will happen to intensities of image(s) and overlay(s)
        if false, dynamic range will be maximized when visualizing overlays
        if 2-tuple, the image will be dynamically scaled between these quantiles

    title : string
        add a title to the plot

    filename : string
        if given, the resulting image will be saved to this file

    dpi : integer
        determines resolution of image if saved to file. Higher values
        result in higher resolution images, but at a cost of having a
        larger file size

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> ants.plot_ortho(mni, xyz=(100,100,100))
    >>> mni2 = mni.threshold_image(7000, mni.max())
    >>> ants.plot_ortho(mni, overlay=mni2)
    >>> ants.plot_ortho(mni, overlay=mni2, flat=True)
    >>> ants.plot_ortho(mni, overlay=mni2, xyz=(110,110,110), xyz_lines=False,
                        text='Lines Turned Off', textfontsize=22)
    >>> ants.plot_ortho(mni, mni2, xyz=(120,100,100),
                        text=' Example \nOrtho Text', textfontsize=26,
                        title='Example Ortho Title', titlefontsize=26)
    """

    def mirror_matrix(x):
        return x[::-1, :]

    def rotate270_matrix(x):
        return mirror_matrix(x.T)

    def reorient_slice(x, axis):
        return rotate270_matrix(x)

    # need this hack because of a weird NaN warning from matplotlib with overlays
    warnings.simplefilter("ignore")

    # handle `image` argument
    if isinstance(image, str):
        image = iio2.image_read(image)
    if not isinstance(image, iio.ANTsImage):
        raise ValueError("image argument must be an ANTsImage")
    if image.dimension != 3:
        raise ValueError("Input image must have 3 dimensions!")

    # handle `overlay` argument
    if overlay is not None:
        vminol = overlay.min()
        vmaxol = overlay.max()
        if isinstance(overlay, str):
            overlay = iio2.image_read(overlay)
        if not isinstance(overlay, iio.ANTsImage):
            raise ValueError("overlay argument must be an ANTsImage")
        if overlay.dimension != 3:
            raise ValueError("Overlay image must have 3 dimensions!")

        if not iio.image_physical_space_consistency(image, overlay):
            overlay = reg.resample_image_to_target(overlay, image, interp_type="linear")

    if blend:
        if alpha == 1:
            alpha = 0.5
        image = image * alpha + overlay * (1 - alpha)
        overlay = None
        alpha = 1.0

    if image.pixeltype not in {"float", "double"}:
        scale = False  # turn off scaling if image is discrete

    # reorient images
    if reorient != False:
        if reorient == True:
            reorient = "RPI"
        image = image.reorient_image2("RPI")
        if overlay is not None:
            overlay = overlay.reorient_image2("RPI")

    # handle `slices` argument
    if xyz is None:
        xyz = [int(s / 2) for s in image.shape]
    for i in range(3):
        if xyz[i] is None:
            xyz[i] = int(image.shape[i] / 2)

    # resample image if spacing is very unbalanced
    spacing = [s for i, s in enumerate(image.spacing)]
    if (max(spacing) / min(spacing)) > 3.0:
        new_spacing = (1, 1, 1)
        image = image.resample_image(tuple(new_spacing))
        if overlay is not None:
            overlay = overlay.resample_image(tuple(new_spacing))
        xyz = [
            int(sl * (sold / snew)) for sl, sold, snew in zip(xyz, spacing, new_spacing)
        ]

    # potentially crop image
    if crop:
        plotmask = image.get_mask(cleanup=0)
        if plotmask.max() == 0:
            plotmask += 1
        image = image.crop_image(plotmask)
        if overlay is not None:
            overlay = overlay.crop_image(plotmask)

    # pad images
    image, lowpad, uppad = image.pad_image(return_padvals=True)
    xyz = [v + l for v, l in zip(xyz, lowpad)]
    if overlay is not None:
        overlay = overlay.pad_image()

    # handle `domain_image_map` argument
    if domain_image_map is not None:
        if isinstance(domain_image_map, iio.ANTsImage):
            tx = tio2.new_ants_transform(
                precision="float",
                transform_type="AffineTransform",
                dimension=image.dimension,
            )
            image = tio.apply_ants_transform_to_image(tx, image, domain_image_map)
            if overlay is not None:
                overlay = tio.apply_ants_transform_to_image(
                    tx, overlay, domain_image_map, interpolation="linear"
                )
        elif isinstance(domain_image_map, (list, tuple)):
            # expect an image and transformation
            if len(domain_image_map) != 2:
                raise ValueError("domain_image_map list or tuple must have length == 2")

            dimg = domain_image_map[0]
            if not isinstance(dimg, iio.ANTsImage):
                raise ValueError("domain_image_map first entry should be ANTsImage")

            tx = domain_image_map[1]
            image = reg.apply_transforms(dimg, image, transform_list=tx)
            if overlay is not None:
                overlay = reg.apply_transforms(
                    dimg, overlay, transform_list=tx, interpolator="linear"
                )

    ## single-channel images ##
    if image.components == 1:

        # potentially find dynamic range
        if scale == True:
            vmin, vmax = image.quantile((0.05, 0.95))
        elif isinstance(scale, (list, tuple)):
            if len(scale) != 2:
                raise ValueError(
                    "scale argument must be boolean or list/tuple with two values"
                )
            vmin, vmax = image.quantile(scale)
        else:
            vmin = None
            vmax = None

        if not flat:
            nrow = 2
            ncol = 2
        else:
            nrow = 1
            ncol = 3

        fig = plt.figure(figsize=(9 * figsize, 9 * figsize))
        if title is not None:
            basey = 0.88 if not flat else 0.66
            basex = 0.5
            fig.suptitle(
                title, fontsize=titlefontsize, x=basex + title_dx, y=basey + title_dy
            )

        gs = gridspec.GridSpec(
            nrow,
            ncol,
            wspace=0.0,
            hspace=0.0,
            top=1.0 - 0.5 / (nrow + 1),
            bottom=0.5 / (nrow + 1),
            left=0.5 / (ncol + 1),
            right=1 - 0.5 / (ncol + 1),
        )

        # pad image to have isotropic array dimensions
        image = image.numpy()
        if overlay is not None:
            overlay = overlay.numpy()
            if overlay.dtype not in ["uint8", "uint32"]:
                overlay[np.abs(overlay) == 0] = np.nan

        yz_slice = reorient_slice(image[xyz[0], :, :], 0)
        ax = plt.subplot(gs[0, 0])
        ax.imshow(yz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            yz_overlay = reorient_slice(overlay[xyz[0], :, :], 0)
            ax.imshow(yz_overlay, alpha=overlay_alpha, cmap=overlay_cmap, vmin=vminol, vmax=vmaxol )
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [yz_slice.shape[0] - xyz[1], yz_slice.shape[0] - xyz[1]],
                [xyz_pad, yz_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, yz_slice.shape[1] - xyz_pad],
                [yz_slice.shape[1] - xyz[2], yz_slice.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        if orient_labels:
            ax.text(
                0.5,
                0.98,
                "S",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.02,
                "I",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.98,
                0.5,
                "A",
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.5,
                "P",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
        ax.axis("off")

        xz_slice = reorient_slice(image[:, xyz[1], :], 1)
        ax = plt.subplot(gs[0, 1])
        ax.imshow(xz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            xz_overlay = reorient_slice(overlay[:, xyz[1], :], 1)
            ax.imshow(xz_overlay, alpha=overlay_alpha, cmap=overlay_cmap, vmin=vminol, vmax=vmaxol )
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xz_slice.shape[0] - xyz[0], xz_slice.shape[0] - xyz[0]],
                [xyz_pad, xz_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xz_slice.shape[1] - xyz_pad],
                [xz_slice.shape[1] - xyz[2], xz_slice.shape[1] - xyz[2]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        if orient_labels:
            ax.text(
                0.5,
                0.98,
                "S",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.02,
                "I",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.98,
                0.5,
                "L",
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.5,
                "R",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
        ax.axis("off")

        xy_slice = reorient_slice(image[:, :, xyz[2]], 2)
        if not flat:
            ax = plt.subplot(gs[1, 1])
        else:
            ax = plt.subplot(gs[0, 2])
        ax.imshow(xy_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            xy_overlay = reorient_slice(overlay[:, :, xyz[2]], 2)
            ax.imshow(xy_overlay, alpha=overlay_alpha, cmap=overlay_cmap, vmin=vminol, vmax=vmaxol )
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xy_slice.shape[0] - xyz[0], xy_slice.shape[0] - xyz[0]],
                [xyz_pad, xy_slice.shape[0] - xyz_pad],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
            l = mlines.Line2D(
                [xyz_pad, xy_slice.shape[1] - xyz_pad],
                [xy_slice.shape[1] - xyz[1], xy_slice.shape[1] - xyz[1]],
                color=xyz_color,
                alpha=xyz_alpha,
                linewidth=xyz_linewidth,
            )
            ax.add_line(l)
        if orient_labels:
            ax.text(
                0.5,
                0.98,
                "A",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.02,
                "P",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.98,
                0.5,
                "L",
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.02,
                0.5,
                "R",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
        ax.axis("off")

        if not flat:
            # empty corner
            ax = plt.subplot(gs[1, 0])
            if text is not None:
                # add text
                left, width = 0.25, 0.5
                bottom, height = 0.25, 0.5
                right = left + width
                top = bottom + height
                ax.text(
                    0.5 * (left + right) + text_dx,
                    0.5 * (bottom + top) + text_dy,
                    text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=textfontsize,
                    color=textfontcolor,
                    transform=ax.transAxes,
                )
            # ax.text(0.5, 0.5)
            ax.imshow(np.zeros(image.shape[:-1]), cmap="Greys_r")
            ax.axis("off")

    ## multi-channel images ##
    elif image.components > 1:
        raise ValueError("Multi-channel images not currently supported!")

    if filename is not None:
        plt.savefig(filename, dpi=dpi, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter("default")


def plot(
    image,
    overlay=None,
    blend=False,
    alpha=1,
    cmap="Greys_r",
    overlay_cmap="turbo",
    overlay_alpha=0.9,
    vminol=None,
    vmaxol=None,
    cbar=False,
    cbar_length=0.8,
    cbar_dx=0.0,
    cbar_vertical=True,
    axis=0,
    nslices=12,
    slices=None,
    ncol=None,
    slice_buffer=None,
    black_bg=True,
    bg_thresh_quant=0.01,
    bg_val_quant=0.99,
    domain_image_map=None,
    crop=False,
    scale=False,
    reverse=False,
    title=None,
    title_fontsize=20,
    title_dx=0.0,
    title_dy=0.0,
    filename=None,
    dpi=500,
    figsize=1.5,
    reorient=True,
    resample=True,
):
    """
    Plot an ANTsImage.

    Use mask_image and/or threshold_image to preprocess images to be be
    overlaid and display the overlays in a given range. See the wiki examples.

    By default, images will be reoriented to 'LAI' orientation before plotting.
    So, if axis == 0, the images will be ordered from the
    left side of the brain to the right side of the brain. If axis == 1,
    the images will be ordered from the anterior (front) of the brain to
    the posterior (back) of the brain. And if axis == 2, the images will
    be ordered from the inferior (bottom) of the brain to the superior (top)
    of the brain.

    ANTsR function: `plot.antsImage`

    Arguments
    ---------
    image : ANTsImage
        image to plot

    overlay : ANTsImage
        image to overlay on base image

    cmap : string
        colormap to use for base image. See matplotlib.

    overlay_cmap : string
        colormap to use for overlay images, if applicable. See matplotlib.

    overlay_alpha : float
        level of transparency for any overlays. Smaller value means
        the overlay is more transparent. See matplotlib.

    axis : integer
        which axis to plot along if image is 3D

    nslices : integer
        number of slices to plot if image is 3D

    slices : list or tuple of integers
        specific slice indices to plot if image is 3D.
        If given, this will override `nslices`.
        This can be absolute array indices (e.g. (80,100,120)), or
        this can be relative array indices (e.g. (0.4,0.5,0.6))

    ncol : integer
        Number of columns to have on the plot if image is 3D.

    slice_buffer : integer
        how many slices to buffer when finding the non-zero slices of
        a 3D images. So, if slice_buffer = 10, then the first slice
        in a 3D image will be the first non-zero slice index plus 10 more
        slices.

    black_bg : boolean
        if True, the background of the image(s) will be black.
        if False, the background of the image(s) will be determined by the
            values `bg_thresh_quant` and `bg_val_quant`.

    bg_thresh_quant : float
        if white_bg=True, the background will be determined by thresholding
        the image at the `bg_thresh` quantile value and setting the background
        intensity to the `bg_val` quantile value.
        This value should be in [0, 1] - somewhere around 0.01 is recommended.
            - equal to 1 will threshold the entire image
            - equal to 0 will threshold none of the image

    bg_val_quant : float
        if white_bg=True, the background will be determined by thresholding
        the image at the `bg_thresh` quantile value and setting the background
        intensity to the `bg_val` quantile value.
        This value should be in [0, 1]
            - equal to 1 is pure white
            - equal to 0 is pure black
            - somewhere in between is gray

    domain_image_map : ANTsImage
        this input ANTsImage or list of ANTsImage types contains a reference image
        `domain_image` and optional reference mapping named `domainMap`.
        If supplied, the image(s) to be plotted will be mapped to the domain
        image space before plotting - useful for non-standard image orientations.

    crop : boolean
        if true, the image(s) will be cropped to their bounding boxes, resulting
        in a potentially smaller image size.
        if false, the image(s) will not be cropped

    scale : boolean or 2-tuple
        if true, nothing will happen to intensities of image(s) and overlay(s)
        if false, dynamic range will be maximized when visualizing overlays
        if 2-tuple, the image will be dynamically scaled between these quantiles

    reverse : boolean
        if true, the order in which the slices are plotted will be reversed.
        This is useful if you want to plot from the front of the brain first
        to the back of the brain, or vice-versa

    title : string
        add a title to the plot

    filename : string
        if given, the resulting image will be saved to this file

    dpi : integer
        determines resolution of image if saved to file. Higher values
        result in higher resolution images, but at a cost of having a
        larger file size

    resample : bool
        if true, resample image if spacing is very unbalanced.

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> segs = img.kmeans_segmentation(k=3)['segmentation']
    >>> ants.plot(img, segs*(segs==1), crop=True)
    >>> ants.plot(img, segs*(segs==1), crop=False)
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> segs = mni.kmeans_segmentation(k=3)['segmentation']
    >>> ants.plot(mni, segs*(segs==1), crop=False)
    """
    if (axis == "x") or (axis == "saggittal"):
        axis = 0
    if (axis == "y") or (axis == "coronal"):
        axis = 1
    if (axis == "z") or (axis == "axial"):
        axis = 2

    def mirror_matrix(x):
        return x[::-1, :]

    def rotate270_matrix(x):
        return mirror_matrix(x.T)

    def rotate180_matrix(x):
        return x[::-1, ::-1]

    def rotate90_matrix(x):
        return x.T

    def flip_matrix(x):
        return mirror_matrix(rotate180_matrix(x))

    def reorient_slice(x, axis):
        if axis != 2:
            x = rotate90_matrix(x)
        if axis == 2:
            x = rotate270_matrix(x)
        x = mirror_matrix(x)
        return x

    # need this hack because of a weird NaN warning from matplotlib with overlays
    warnings.simplefilter("ignore")

    # handle `image` argument
    if isinstance(image, str):
        image = iio2.image_read(image)
    if not isinstance(image, iio.ANTsImage):
        raise ValueError("image argument must be an ANTsImage")

    assert image.sum() > 0, "Image must be non-zero"

    if (image.pixeltype not in {"float", "double"}) or (image.is_rgb):
        scale = False  # turn off scaling if image is discrete

    # handle `overlay` argument
    if overlay is not None:
        if vminol is None:
            vminol = overlay.min()
        if vmaxol is None:
            vmaxol = overlay.max()
        if isinstance(overlay, str):
            overlay = iio2.image_read(overlay)
        if not isinstance(overlay, iio.ANTsImage):
            raise ValueError("overlay argument must be an ANTsImage")

        if not iio.image_physical_space_consistency(image, overlay):
            overlay = reg.resample_image_to_target(overlay, image, interp_type="nearestNeighbor")

        if blend:
            if alpha == 1:
                alpha = 0.5
            image = image * alpha + overlay * (1 - alpha)
            overlay = None
            alpha = 1.0

    # handle `domain_image_map` argument
    if domain_image_map is not None:
        if isinstance(domain_image_map, iio.ANTsImage):
            tx = tio2.new_ants_transform(
                precision="float",
                transform_type="AffineTransform",
                dimension=image.dimension,
            )
            image = tio.apply_ants_transform_to_image(tx, image, domain_image_map)
            if overlay is not None:
                overlay = tio.apply_ants_transform_to_image(
                    tx, overlay, domain_image_map, interpolation="nearestNeighbor"
                )
        elif isinstance(domain_image_map, (list, tuple)):
            # expect an image and transformation
            if len(domain_image_map) != 2:
                raise ValueError("domain_image_map list or tuple must have length == 2")

            dimg = domain_image_map[0]
            if not isinstance(dimg, iio.ANTsImage):
                raise ValueError("domain_image_map first entry should be ANTsImage")

            tx = domain_image_map[1]
            image = reg.apply_transforms(dimg, image, transform_list=tx)
            if overlay is not None:
                overlay = reg.apply_transforms(
                    dimg, overlay, transform_list=tx, interpolator="linear"
                )

    ## single-channel images ##
    if image.components == 1:

        # potentially crop image
        if crop:
            plotmask = image.get_mask(cleanup=0)
            if plotmask.max() == 0:
                plotmask += 1
            image = image.crop_image(plotmask)
            if overlay is not None:
                overlay = overlay.crop_image(plotmask)

        # potentially find dynamic range
        if scale == True:
            vmin, vmax = image.quantile((0.05, 0.95))
        elif isinstance(scale, (list, tuple)):
            if len(scale) != 2:
                raise ValueError(
                    "scale argument must be boolean or list/tuple with two values"
                )
            vmin, vmax = image.quantile(scale)
        else:
            vmin = None
            vmax = None

        # Plot 2D image
        if image.dimension == 2:

            img_arr = image.numpy()
            img_arr = rotate90_matrix(img_arr)

            if not black_bg:
                img_arr[img_arr < image.quantile(bg_thresh_quant)] = image.quantile(
                    bg_val_quant
                )

            if overlay is not None:
                ov_arr = overlay.numpy()
                ov_arr = rotate90_matrix(ov_arr)
                if ov_arr.dtype not in ["uint8", "uint32"]:
                    ov_arr = np.ma.masked_where(ov_arr == 0, ov_arr)

            fig = plt.figure()
            if title is not None:
                fig.suptitle(
                    title, fontsize=title_fontsize, x=0.5 + title_dx, y=0.95 + title_dy
                )

            ax = plt.subplot(111)

            # plot main image
            im = ax.imshow(img_arr, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)

            if overlay is not None:
                im = ax.imshow(ov_arr, alpha=overlay_alpha, cmap=overlay_cmap,
                    vmin=vminol, vmax=vmaxol )

            if cbar:
                cbar_orient = "vertical" if cbar_vertical else "horizontal"
                fig.colorbar(im, orientation=cbar_orient)

            plt.axis("off")

        # Plot 3D image
        elif image.dimension == 3:
            # resample image if spacing is very unbalanced
            spacing = [s for i, s in enumerate(image.spacing) if i != axis]
            was_resampled = False
            if (max(spacing) / min(spacing)) > 3.0 and resample:
                was_resampled = True
                new_spacing = (1, 1, 1)
                image = image.resample_image(tuple(new_spacing))
                if overlay is not None:
                    overlay = overlay.resample_image(tuple(new_spacing))

            if reorient:
                image = image.reorient_image2("LAI")
            img_arr = image.numpy()
            # reorder dims so that chosen axis is first
            img_arr = np.rollaxis(img_arr, axis)

            if overlay is not None:
                if reorient:
                    overlay = overlay.reorient_image2("LAI")
                ov_arr = overlay.numpy()
                if ov_arr.dtype not in ["uint8", "uint32"]:
                    ov_arr = np.ma.masked_where(ov_arr == 0, ov_arr)
                ov_arr = np.rollaxis(ov_arr, axis)

            if slices is None:
                if not isinstance(slice_buffer, (list, tuple)):
                    if slice_buffer is None:
                        slice_buffer = (
                            int(img_arr.shape[1] * 0.1),
                            int(img_arr.shape[2] * 0.1),
                        )
                    else:
                        slice_buffer = (slice_buffer, slice_buffer)
                nonzero = np.where(img_arr.sum(axis=(1, 2)) > 0.01)[0]
                min_idx = nonzero[0] + slice_buffer[0]
                max_idx = nonzero[-1] - slice_buffer[1]
                slice_idxs = np.linspace(min_idx, max_idx, nslices).astype("int")
                if reverse:
                    slice_idxs = np.array(list(reversed(slice_idxs)))
            else:
                if isinstance(slices, (int, float)):
                    slices = [slices]
                # if all slices are less than 1, infer that they are relative slices
                if sum([s > 1 for s in slices]) == 0:
                    slices = [int(s * img_arr.shape[0]) for s in slices]
                slice_idxs = slices
                nslices = len(slices)

            if was_resampled:
                # re-calculate slices to account for new image shape
                slice_idxs = np.unique(
                    np.array(
                        [
                            int(s * (image.shape[axis] / img_arr.shape[0]))
                            for s in slice_idxs
                        ]
                    )
                )

            # only have one row if nslices <= 6 and user didnt specify ncol
            if ncol is None:
                if nslices <= 6:
                    ncol = nslices
                else:
                    ncol = int(round(math.sqrt(nslices)))

            # calculate grid size
            nrow = math.ceil(nslices / ncol)
            xdim = img_arr.shape[2]
            ydim = img_arr.shape[1]

            dim_ratio = ydim / xdim
            fig = plt.figure(
                figsize=((ncol + 1) * figsize * dim_ratio, (nrow + 1) * figsize)
            )
            if title is not None:
                fig.suptitle(
                    title, fontsize=title_fontsize, x=0.5 + title_dx, y=0.95 + title_dy
                )

            gs = gridspec.GridSpec(
                nrow,
                ncol,
                wspace=0.0,
                hspace=0.0,
                top=1.0 - 0.5 / (nrow + 1),
                bottom=0.5 / (nrow + 1),
                left=0.5 / (ncol + 1),
                right=1 - 0.5 / (ncol + 1),
            )

            slice_idx_idx = 0
            for i in range(nrow):
                for j in range(ncol):
                    if slice_idx_idx < len(slice_idxs):
                        imslice = img_arr[slice_idxs[slice_idx_idx]]
                        imslice = reorient_slice(imslice, axis)
                        if not black_bg:
                            imslice[
                                imslice < image.quantile(bg_thresh_quant)
                            ] = image.quantile(bg_val_quant)
                    else:
                        imslice = np.zeros_like(img_arr[0])
                        imslice = reorient_slice(imslice, axis)

                    ax = plt.subplot(gs[i, j])
                    im = ax.imshow(imslice, cmap=cmap, vmin=vmin, vmax=vmax)

                    if overlay is not None:
                        if slice_idx_idx < len(slice_idxs):
                            ovslice = ov_arr[slice_idxs[slice_idx_idx]]
                            ovslice = reorient_slice(ovslice, axis)
                            im = ax.imshow(
                                ovslice, alpha=overlay_alpha, cmap=overlay_cmap,
                                    vmin=vminol, vmax=vmaxol )
                    ax.axis("off")
                    slice_idx_idx += 1

            if cbar:
                cbar_start = (1 - cbar_length) / 2
                if cbar_vertical:
                    cax = fig.add_axes([0.9 + cbar_dx, cbar_start, 0.03, cbar_length])
                    cbar_orient = "vertical"
                else:
                    cax = fig.add_axes([cbar_start, 0.08 + cbar_dx, cbar_length, 0.03])
                    cbar_orient = "horizontal"
                fig.colorbar(im, cax=cax, orientation=cbar_orient)

    ## multi-channel images ##
    elif image.components > 1:
        if not image.is_rgb:
            if not image.components == 3:
                raise ValueError("Multi-component images only supported if they have 3 components")

        img_arr = image.numpy()
        img_arr = img_arr / img_arr.max()
        img_arr = np.stack(
            [rotate90_matrix(img_arr[:, :, i]) for i in range(3)], axis=-1
        )

        fig = plt.figure()
        ax = plt.subplot(111)

        # plot main image
        ax.imshow(img_arr, alpha=alpha)

        plt.axis("off")

    if filename is not None:
        filename = os.path.expanduser(filename)
        plt.savefig(filename, dpi=dpi, transparent=True, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter("default")


def plot_directory(
    directory,
    recursive=False,
    regex="*",
    save_prefix="",
    save_suffix="",
    axis=None,
    **kwargs
):
    """
    Create and save an ANTsPy plot for every image matching a given regular
    expression in a directory, optionally recursively. This is a good function
    for quick visualize exploration of all of images in a directory

    ANTsR function: N/A

    Arguments
    ---------
    directory : string
        directory in which to search for images and plot them

    recursive : boolean
        If true, this function will search through all directories under
        the given directory recursively to make plots.
        If false, this function will only create plots for images in the
        given directory

    regex : string
        regular expression used to filter out certain filenames or suffixes

    save_prefix : string
        sub-string that will be appended to the beginning of all saved plot filenames.
        Default is to add nothing.

    save_suffix : string
        sub-string that will be appended to the end of all saved plot filenames.
        Default is add nothing.

    kwargs : keyword arguments
        any additional arguments to pass onto the `ants.plot` function.
        e.g. overlay, alpha, cmap, etc. See `ants.plot` for more options.

    Example
    -------
    >>> import ants
    >>> ants.plot_directory(directory='~/desktop/testdir',
                            recursive=False, regex='*')
    """

    def has_acceptable_suffix(fname):
        suffixes = {".nii.gz"}
        return sum([fname.endswith(sx) for sx in suffixes]) > 0

    if directory.startswith("~"):
        directory = os.path.expanduser(directory)

    if not os.path.isdir(directory):
        raise ValueError("directory %s does not exist!" % directory)

    for root, dirnames, fnames in os.walk(directory):
        for fname in fnames:
            if fnmatch.fnmatch(fname, regex) and has_acceptable_suffix(fname):
                load_fname = os.path.join(root, fname)
                fname = fname.replace(".".join(fname.split(".")[1:]), "png")
                fname = fname.replace(".png", "%s.png" % save_suffix)
                fname = "%s%s" % (save_prefix, fname)
                save_fname = os.path.join(root, fname)
                img = iio2.image_read(load_fname)

                if axis is None:
                    axis_range = [i for i in range(img.dimension)]
                else:
                    axis_range = axis if isinstance(axis, (list, tuple)) else [axis]

                if img.dimension > 2:
                    for axis_idx in axis_range:
                        filename = save_fname.replace(".png", "_axis%i.png" % axis_idx)
                        ncol = int(math.sqrt(img.shape[axis_idx]))
                        plot(
                            img,
                            axis=axis_idx,
                            nslices=img.shape[axis_idx],
                            ncol=ncol,
                            filename=filename,
                            **kwargs
                        )
                else:
                    filename = save_fname
                    plot(img, filename=filename, **kwargs)
