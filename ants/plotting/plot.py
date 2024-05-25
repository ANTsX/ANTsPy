"""
Functions for plotting ants images
"""


__all__ = [
    "plot"
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

    def reorient_slice(x, axis):
        if axis != 2:
            x = rotate90_matrix(x)
        if axis == 2:
            x = rotate270_matrix(x)
        x = mirror_matrix(x)
        return x


    # handle `image` argument
    if isinstance(image, str):
        image = ants.image_read(image)
    if not ants.is_image(image):
        raise ValueError("image argument must be an ANTsImage")

    if np.all(np.equal(image.numpy(), 0.0)):
        warnings.warn("Image must be non-zero. will not plot.")
        return

    # need this hack because of a weird NaN warning from matplotlib with overlays
    warnings.simplefilter("ignore")

    if (image.pixeltype not in {"float", "double"}) or (image.is_rgb):
        scale = False  # turn off scaling if image is discrete

    # handle `overlay` argument
    if overlay is not None:
        if isinstance(overlay, str):
            overlay = ants.image_read(overlay)
        if vminol is None:
            vminol = overlay.min()
        if vmaxol is None:
            vmaxol = overlay.max()
        if not ants.is_image(overlay):
            raise ValueError("overlay argument must be an ANTsImage")
        if overlay.components > 1:
            raise ValueError("overlay cannot have more than one voxel component")

        if not ants.image_physical_space_consistency(image, overlay):
            overlay = ants.resample_image_to_target(overlay, image, interp_type="nearestNeighbor")

        if blend:
            if alpha == 1:
                alpha = 0.5
            image = image * alpha + overlay * (1 - alpha)
            overlay = None
            alpha = 1.0

    # handle `domain_image_map` argument
    if domain_image_map is not None:
        tx = ants.new_ants_transform(
            precision="float",
            transform_type="AffineTransform",
            dimension=image.dimension,
        )
        image = ants.apply_ants_transform_to_image(tx, image, domain_image_map)
        if overlay is not None:
            overlay = ants.apply_ants_transform_to_image(
                tx, overlay, domain_image_map, interpolation="nearestNeighbor"
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
                mask = ov_arr == 0
                mask = np.ma.masked_where(mask == 0, mask)
                ov_arr = np.ma.masked_array(ov_arr, mask)
                ov_arr = rotate90_matrix(ov_arr)

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
                mask = ov_arr == 0
                mask = np.ma.masked_where(mask == 0, mask)
                ov_arr = np.ma.masked_array(ov_arr, mask)
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
                if min_idx > max_idx:
                    temp = min_idx
                    min_idx = max_idx
                    max_idx = temp
                if max_idx > nonzero.max():
                    max_idx = nonzero.max()
                if min_idx < 0:
                    min_idx = 0
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
    elif image.has_components:
        raise Exception('Plotting images with components is not currently supported.')

    if filename is not None:
        filename = os.path.expanduser(filename)
        plt.savefig(filename, dpi=dpi, transparent=True, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter("default")


