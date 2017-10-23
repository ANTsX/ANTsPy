"""
Create a static 2D image of a 2D ANTsImage
or a tile of slices from a 3D ANTsImage

TODO:
- add `ortho` function for plotting 3d ortho slices
- add `plot_multichannel` function for plotting multi-channel images
    - support for quivers as well
- add `plot_grid` function for plotting slices in arbitrary grids
"""


__all__ = ['plot',
           'plot_directory']

import fnmatch
import math
import os
import warnings

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from .. import registration as reg
from ..core import ants_image as iio
from ..core import ants_image_io as iio2
from ..core import ants_transform as tio
from ..core import ants_transform_io as tio2



def plot(image, overlay=None, cmap='Greys_r', alpha=1, overlay_cmap='jet', overlay_alpha=0.9,
         axis=0, nslices=12, slices=None, ncol=4, slice_buffer=0, black_bg=True,
         bg_thresh_quant=0.01, bg_val_quant=1.0, domain_image_map=None, crop=False, scale=True,
         title=None, filename=None):
    """
    Plot an ANTsImage
    
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

    scale : boolean
        if true, nothing will happen to intensities of image(s) and overlay(s)
        if false, dynamic range will be maximized when visualizing overlays
    
    title : string 
        add a title to the plot

    filename : string
        if given, the resulting image will be saved to this file

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
    def mirror_matrix(x):
        return x[::-1,:]
    def rotate270_matrix(x):
        return mirror_matrix(x.T)
    def rotate180_matrix(x):
        return x[::-1,:]
    def rotate90_matrix(x):
        return mirror_matrix(x).T
    def flip_matrix(x):
        return mirror_matrix(rotate180_matrix(x))
    def reorient_slice(x, axis):
        if (axis != 1):
            x = rotate90_matrix(x)
        if (axis == 1):
            x = rotate90_matrix(x)
        x = mirror_matrix(x)
        return x
    # need this hack because of a weird NaN warning from matplotlib with overlays
    warnings.simplefilter('ignore')

    # handle `image` argument
    if isinstance(image, str):
        image = iio2.image_read(image)
    if not isinstance(image, iio.ANTsImage):
        raise ValueError('image argument must be an ANTsImage')
    
    if image.pixeltype not in {'float', 'double'}:
        scale = False # turn off scaling if image is discrete

    # handle `overlay` argument 
    if overlay is not None:
        if isinstance(overlay, str):
            overlay = iio2.image_read(overlay)
        if not isinstance(overlay, iio.ANTsImage):
            raise ValueError('overlay argument must be an ANTsImage')

        if not iio.image_physical_space_consistency(image, overlay):
            overlay = reg.resample_image_to_target(overlay, image, interp_type='linear')

    # handle `domain_image_map` argument
    if domain_image_map is not None:
        if isinstance(domain_image_map, iio.ANTsImage):
            tx = tio2.new_ants_transform(precision='float', transform_type='AffineTransform',
                                         dimension=image.dimension)
            image = tio.apply_ants_transform_to_image(tx, image, domain_image_map)
            if overlay is not None:
                overlay = tio.apply_ants_transform_to_image(tx, overlay, 
                                                            domain_image_map, 
                                                            interpolation='linear')
        elif isinstance(domain_image_map, (list, tuple)):
            # expect an image and transformation
            if len(domain_image_map) != 2:
                raise ValueError('domain_image_map list or tuple must have length == 2')
            
            dimg = domain_image_map[0]
            if not isinstance(dimg, iio.ANTsImage):
                raise ValueError('domain_image_map first entry should be ANTsImage')

            tx = domain_image_map[1]
            image = reg.apply_transforms(dimg, image, transform_list=tx)
            if overlay is not None:
                overlay = reg.apply_transforms(dimg, overlay, transform_list=tx,
                                               interpolator='linear')

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
            vmin, vmax = image.quantile((0.05,0.95))
        elif isinstance(scale, (list,tuple)):
            if len(scale) != 2:
                raise ValueError('scale argument must be boolean or list/tuple with two values')
            vmin, vmax = scale
        else:
            vmin = None
            vmax = None

        # Plot 2D image
        if image.dimension == 2:

            img_arr = image.numpy()
            img_arr = rotate90_matrix(img_arr)

            if white_bg:
                img_arr[img_arr<image.quantile(bg_thresh_quant)] = image.quantile(bg_val_quant)

            if overlay is not None:
                ov_arr = overlay.numpy()
                ov_arr = rotate90_matrix(ov_arr)
                ov_arr[np.abs(ov_arr) == 0] = np.nan

            fig, ax = plt.subplots()

            # plot main image
            ax.imshow(img_arr, cmap=cmap,
                      alpha=alpha, 
                      vmin=vmin, vmax=vmax)

            if overlay is not None:

                ax.imshow(ov_arr, 
                          alpha=overlay_alpha, 
                          cmap=overlay_cmap)

            plt.axis('off')
            if filename is not None:
                plt.savefig(filename)
                plt.close(fig)
            else:
                plt.show()

        # Plot 3D image
        elif image.dimension == 3:
            img_arr = image.numpy()
            # reorder dims so that chosen axis is first
            img_arr = np.rollaxis(img_arr, axis)

            if overlay is not None:
                ov_arr = overlay.numpy()
                ov_arr[np.abs(ov_arr) == 0] = np.nan
                ov_arr = np.rollaxis(ov_arr, axis)

            if slices is None:
                if not isinstance(slice_buffer, (list, tuple)):
                    slice_buffer = (slice_buffer, slice_buffer)
                nonzero = np.where(np.abs(img_arr)>0)[0]
                min_idx = nonzero[0] + slice_buffer[0]
                max_idx = nonzero[-1] - slice_buffer[1]
                slice_idxs = np.linspace(min_idx, max_idx, nslices).astype('int')
            else:
                if isinstance(slices, (int,float)):
                    slices = [slices]
                # if all slices are less than 1, infer that they are relative slices
                if sum([s > 1 for s in slices]) == 0:
                    slices = [int(s*img_arr.shape[0]) for s in slices]
                slice_idxs = slices
                nslices = len(slices)

            # only have one row if nslices <= 6 and user didnt specify ncol
            if (nslices <= 6) and (ncol==4):
                ncol = nslices

            # calculate grid size
            nrow = math.ceil(nslices / ncol)

            xdim = img_arr.shape[2]
            ydim = img_arr.shape[1]

            fig = plt.figure(figsize=((ncol+1)*1.5*(ydim/xdim), (nrow+1)*1.5)) 

            gs = gridspec.GridSpec(nrow, ncol,
                     wspace=0.0, hspace=0.0, 
                     top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                     left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

            slice_idx_idx = 0
            for i in range(nrow):
                for j in range(ncol):
                    if slice_idx_idx < len(slice_idxs):
                        imslice = img_arr[slice_idxs[slice_idx_idx]]
                        imslice = reorient_slice(imslice, axis)
                        if white_bg:
                            imslice[imslice<image.quantile(bg_thresh_quant)] = image.quantil(bg_val_quant)
                    else:
                        imslice = np.zeros_like(img_arr[0])

                    ax = plt.subplot(gs[i,j])
                    ax.imshow(imslice, cmap=cmap,
                              vmin=vmin, vmax=vmax)

                    if overlay is not None:
                        if slice_idx_idx < len(slice_idxs):
                            ovslice = ov_arr[slice_idxs[slice_idx_idx]]
                            ovslice = reorient_slice(ovslice, axis)
                            ax.imshow(ovslice, alpha=overlay_alpha, cmap=overlay_cmap)
                    ax.axis('off')
                    slice_idx_idx += 1
    
    ## multi-channel images ##
    elif image.components > 1:
        raise ValueError('Multi-channel images not currently supported!')

    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter('default')


def plot_directory(directory, recursive=False, regex='*', 
                   save_prefix='', save_suffix='', axis=None, **kwargs):
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
        suffixes = {'.nii.gz'}
        return sum([fname.endswith(sx) for sx in suffixes]) > 0

    if directory.startswith('~'):
        directory = os.path.expanduser(directory)

    if not os.path.isdir(directory):
        raise ValueError('directory %s does not exist!' % directory)

    for root, dirnames, fnames in os.walk(directory):
        for fname in fnames:
            if fnmatch.fnmatch(fname, regex) and has_acceptable_suffix(fname):
                load_fname = os.path.join(root, fname)
                fname = fname.replace('.'.join(fname.split('.')[1:]), 'png')
                fname = fname.replace('.png', '%s.png' % save_suffix)
                fname = '%s%s' % (save_prefix, fname)
                save_fname = os.path.join(root, fname)
                img = iio2.image_read(load_fname)

                if axis is None:
                    axis_range = [i for i in range(img.dimension)]
                else:
                    axis_range = axis if isinstance(axis,(list,tuple)) else [axis]

                if img.dimension > 2:
                    for axis_idx in axis_range:
                        filename = save_fname.replace('.png', '_axis%i.png' % axis_idx)
                        ncol = int(math.sqrt(img.shape[axis_idx]))
                        plot(img, axis=axis_idx, nslices=img.shape[axis_idx], ncol=ncol,
                             filename=filename, **kwargs)
                else:
                    filename = save_fname
                    plot(img, filename=filename, **kwargs)                    







