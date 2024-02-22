"""
Functions for plotting ants images
"""


__all__ = [
    "plot_ortho_double"
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
        if overlay.components > 1:
            raise ValueError("overlay cannot have more than one voxel component")
        if overlay.dimension != 3:
            raise ValueError("Overlay image must have 3 dimensions!")

        if not iio.image_physical_space_consistency(image, overlay):
            overlay = reg.resample_image_to_target(overlay, image, interp_type="linear")

    if overlay2 is not None:
        if isinstance(overlay2, str):
            overlay2 = iio2.image_read(overlay2)
        if not isinstance(overlay2, iio.ANTsImage):
            raise ValueError("overlay2 argument must be an ANTsImage")
        if overlay2.components > 1:
            raise ValueError("overlay2 cannot have more than one voxel component")
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
