"""
Functions for plotting ants images
"""


__all__ = [
    "plot_ortho_stack"
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
import ants




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
            images[i] = ants.image_read(images[i])
        if not ants.is_image(images[i]):
            raise ValueError("image argument must be an ANTsImage")
        if images[i].dimension != 3:
            raise ValueError("Input image must have 3 dimensions!")

    if overlays is None:
        overlays = [None] * n_images
    # handle `overlay` argument
    for i in range(n_images):
        if overlays[i] is not None:
            if isinstance(overlays[i], str):
                overlays[i] = ants.image_read(overlays[i])
            if not ants.is_image(overlays[i]):
                raise ValueError("overlay argument must be an ANTsImage")
            if overlays[i].components > 1:
                raise ValueError("overlays[i] cannot have more than one voxel component")
            if overlays[i].dimension != 3:
                raise ValueError("Overlay image must have 3 dimensions!")

            if not ants.image_physical_space_consistency(images[i], overlays[i]):
                overlays[i] = ants.resample_image_to_target(
                    overlays[i], images[i], interp_type="linear"
                )

    for i in range(1, n_images):
        if not ants.image_physical_space_consistency(images[0], images[i]):
            images[i] = ants.resample_image_to_target(
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
        if ants.is_image(domain_image_map):
            tx = ants.new_ants_transform(
                precision="float", transform_type="AffineTransform", dimension=3
            )
            for i in range(n_images):
                images[i] = ants.apply_ants_transform_to_image(
                    tx, images[i], domain_image_map
                )

                if overlays[i] is not None:
                    overlays[i] = ants.apply_ants_transform_to_image(
                        tx, overlays[i], domain_image_map, interpolation="linear"
                    )
        else:
            raise Exception('The domain_image_map must be an ants image.')

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
            title, fontsize=titlefontsize, color=textfontcolor, x=basex + title_dx, y=basey + title_dy
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
                "I",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=20 * figsize,
                color=textfontcolor,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.02,
                "S",
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
