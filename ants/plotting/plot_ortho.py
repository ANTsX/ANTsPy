"""
Functions for plotting ants images
"""


__all__ = [
    "plot_ortho"
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
from ants.decorators import image_method

@image_method
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
    cbar=False,
    cbar_length=0.8,
    cbar_dx=0.0,
    cbar_vertical=True,
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
    resample=False,
    allow_xyz_change=True,
):
    """
    Plot an orthographic view of a 3D image

    Use mask_image and/or threshold_image to preprocess images to be be
    overlaid and display the overlays in a given range. See the wiki examples.

    ANTsR function: N/A

    Arguments
    ---------
    image : ANTsImage
        image to plot

    overlay : ANTsImage
        image to overlay on base image

    xyz : list or tuple of 3 integers
        selects index location on which to center display
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

    cbar: boolean
        if true, a colorbar will be added to the plot

    cbar_length: float
        length of the colorbar relative to the image

    cbar_dx: float
        horizontal shift of the colorbar relative to the image

    cbar_vertical: boolean
        if true, the colorbar will be vertical, if false, it will be
        horizontal underneath the image

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

    resample : resample image in case of unbalanced spacing

    allow_xyz_change : boolean will attempt to adjust xyz after padding

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
        image = ants.image_read(image)
    if not ants.is_image(image):
        raise ValueError("image argument must be an ANTsImage")
    if image.dimension != 3:
        raise ValueError("Input image must have 3 dimensions!")

    # handle `overlay` argument
    if overlay is not None:
        if isinstance(overlay, str):
            overlay = ants.image_read(overlay)
        vminol = overlay.min()
        vmaxol = overlay.max()
        if not ants.is_image(overlay):
            raise ValueError("overlay argument must be an ANTsImage")
        if overlay.components > 1:
            raise ValueError("overlay cannot have more than one voxel component")
        if overlay.dimension != 3:
            raise ValueError("Overlay image must have 3 dimensions!")

        if not ants.image_physical_space_consistency(image, overlay):
            overlay = ants.resample_image_to_target(overlay, image, interp_type="linear")

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
    if (max(spacing) / min(spacing)) > 3.0 and resample:
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
    if True:
        image, lowpad, uppad = image.pad_image(return_padvals=True)
        if allow_xyz_change:
            xyz = [v + l for v, l in zip(xyz, lowpad)]
        if overlay is not None:
            overlay = overlay.pad_image()


    # handle `domain_image_map` argument
    if domain_image_map is not None:
        if ants.is_image(domain_image_map):
            tx = ants.new_ants_transform(
                precision="float",
                transform_type="AffineTransform",
                dimension=image.dimension,
            )
            image = ants.apply_ants_transform_to_image(tx, image, domain_image_map)
            if overlay is not None:
                overlay = ants.apply_ants_transform_to_image(
                    tx, overlay, domain_image_map, interpolation="linear"
                )
        else:
            raise Exception('The domain_image_map must be an image.')

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
                title, fontsize=titlefontsize, color=textfontcolor, x=basex + title_dx, y=basey + title_dy
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
        imageReturn = image.clone()
        image = image.numpy()
        overlayReturn = None
        if overlay is not None:
            overlayReturn = overlay.clone()
            overlay = overlay.numpy()
            if overlay.dtype not in ["uint8", "uint32"]:
                overlay = np.ma.masked_where( np.abs(overlay) <= 1e-16, overlay)
#                overlay[np.abs(overlay) == 0] = np.nan

        yz_slice = reorient_slice(image[xyz[0], :, :], 0)
        ax = plt.subplot(gs[0, 0])
        ax.imshow(yz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            yz_overlay = reorient_slice(overlay[xyz[0], :, :], 0)
            ax.imshow(yz_overlay, alpha=overlay_alpha, cmap=overlay_cmap, vmin=vminol, vmax=vmaxol )
        if xyz_lines:
            # add lines
            l = mlines.Line2D(
                [xyz[1], xyz[1]],
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
        im = ax.imshow(xy_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            xy_overlay = reorient_slice(overlay[:, :, xyz[2]], 2)
            im = ax.imshow(xy_overlay, alpha=overlay_alpha, cmap=overlay_cmap, vmin=vminol, vmax=vmaxol)

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
        raise ValueError("Multi-channel images not currently supported!")

    if filename is not None:
        plt.savefig(filename, dpi=dpi, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter("default")

