"""
Create a static 2D image of a 2D ANTsImage
or a tile of slices from a 3D ANTsImage

TODO:
- add `plot_multichannel` function for plotting multi-channel images
    - support for quivers as well

            if colidx == 0:
                left, width = .25, .5
                bottom, height = .25, .5
                right = left + width
                top = bottom + height
                if rlabels[rowidx] is not None:
                    ax.text(-0.07-textpadleft, 0.5*(bottom+top), rlabels[rowidx],
                            horizontalalignment='right',
                            verticalalignment='center',
                            rotation='vertical',
                            bbox={'facecolor':'darkcyan', 'edgecolor':'none',
                                 'alpha':0.9, 'pad':8},
                            transform=ax.transAxes, fontsize=fontsize, color='white',
                            weight=fontweight, size=textsize,
                            path_effects=[path_effects.Stroke(linewidth=3, foreground='black'),
                                          path_effects.Normal()])

              ctextsize=20, cfontsize=14, cfontweight='bold', ctextpad=0., clabels=None,
              cboxstyle='round',
"""


__all__ = ['plot',
           'plot_grid',
           'plot_ortho',
           'plot_directory']

import fnmatch
import math
import os
import warnings

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches

import numpy as np

from .. import registration as reg
from ..core import ants_image as iio
from ..core import ants_image_io as iio2
from ..core import ants_transform as tio
from ..core import ants_transform_io as tio2


def plot_grid(images, slices=None, axes=2, 
              # general figure arguments
              figsize=1., rpad=0, cpad=0,
              # title arguments
              title=None, tfontsize=20, tdx=0, tdy=0,
              # row arguments
              rlabels=None, rfontsize=14, rfontcolor='white', rfacecolor='black', 
              # column arguments 
              clabels=None, cfontsize=14, cfontcolor='white', cfacecolor='black',
              # save arguments
              filename=None, dpi=400, transparent=True,
              # other args
              **kwargs):
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
    >>> ants.plot_grid(images, slices, title='Publication Figures with ANTsPy',
                       tfontsize=20, tdy=0.03, tdx=-0.04,
                       rlabels=['Row 1', 'Row 2'], clabels=['Col 1', 'Col 2'],
                       rfontsize=16, cfontsize=16,
                       filename='/users/ncullen/desktop/img1.png', dpi=600)
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

    if isinstance(images, np.ndarray):
        images = images.tolist()
    if not isinstance(images, list):
        raise ValueError('images argument must be of type list')
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
        rlabels = [None]*nrow
    if clabels is None:
        clabels = [None]*ncol

    if (not one_slice):
        if (nrow != nslicerow) or (ncol != nslicecol):
            raise ValueError('`images` arg shape (%i,%i) must equal `slices` arg shape (%i,%i)!' % (nrow,ncol,nslicerow,nslicecol))

    fig = plt.figure(figsize=((ncol+1)*2.5*figsize, (nrow+1)*2.5*figsize))

    if title is not None:
        basex = 0.5
        basey = 0.9 if clabels[0] is None else 0.95
        fig.suptitle(title, fontsize=tfontsize, x=basex+tdx, y=basey+tdy)

    if (cpad > 0) and (rpad > 0):
        bothgridpad = max(cpad, rpad)
        cpad = 0
        rpad = 0
    else:
        bothgridpad = 0.0

    gs = gridspec.GridSpec(nrow, ncol, wspace=bothgridpad, hspace=0.0,
                 top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1) + cpad, 
                 left=0.5/(ncol+1) + rpad, right=1-0.5/(ncol+1))

    for rowidx in range(nrow):
        for colidx in range(ncol):
            ax = plt.subplot(gs[rowidx, colidx])

            if colidx == 0:
                if rlabels[rowidx] is not None:
                    bottom, height = .25, .5
                    top = bottom + height
                    # add label text
                    ax.text(-0.07, 0.5*(bottom+top), rlabels[rowidx],
                            horizontalalignment='right', verticalalignment='center',
                            rotation='vertical', transform=ax.transAxes,
                            color=rfontcolor, fontsize=rfontsize)

                    # add label background
                    extra = 0.3 if rowidx == 0 else 0.0

                    rect = patches.Rectangle((-0.3, 0), 0.3, 1.0+extra, 
                        facecolor=rfacecolor,
                        alpha=1., transform=ax.transAxes, clip_on=False)
                    ax.add_patch(rect)

            if rowidx == 0:
                if clabels[colidx] is not None:
                    bottom, height = .25, .5
                    left, width = .25, .5
                    right = left + width
                    top = bottom + height
                    ax.text(0.5*(left+right), 0.09+top+bottom, clabels[colidx],
                            horizontalalignment='center', verticalalignment='center',
                            rotation='horizontal', transform=ax.transAxes,
                            color=cfontcolor, fontsize=cfontsize)

                    # add label background
                    rect = patches.Rectangle((0, 1.), 1.0, 0.3, 
                        facecolor=cfacecolor, 
                        alpha=1., transform=ax.transAxes, clip_on=False)
                    ax.add_patch(rect)

            tmpimg = images[rowidx][colidx]
            sliceidx = slices[rowidx][colidx] if not one_slice else slices
            tmpslice = reorient_slice(tmpimg[:,sliceidx,:],0)
            ax.imshow(tmpslice, cmap='Greys_r')
            ax.axis('off')

    if filename is not None:
        filename = os.path.expanduser(filename)
        plt.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def plot_ortho(image, overlay=None, slices=None, xyz=None, flat=False, cmap='Greys_r', alpha=1, 
              overlay_cmap='jet', overlay_alpha=0.9, xyz_cmap='Reds_r', xyz_alpha=1., 
              black_bg=True, bg_thresh_quant=0.01, bg_val_quant=0.99, 
              domain_image_map=None, crop=False, scale=True, title=None, 
              filename=None, dpi=500, figsize=1.):
    """
    Plot an orthographic view of a 3D image
    
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
    >>> ants.plot_ortho(mni, slices=(100,100,100))
    >>> mni2 = mni.threshold_image(7000,mni.max())
    >>> ants.plot_ortho(mni, overlay=mni2)
    >>> ants.plot_ortho(mni, overlay=mni2, flat=True)
    >>> ants.plot_ortho(mni, slices=(100,100,100), xyz=(110,110,110))
    >>> ants.plot_ortho(mni, overlay=mni2, slices=(100,100,100), xyz=(110,110,110))
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
    if image.dimension != 3:
        raise ValueError('Input image must have 3 dimensions!')
    
    if image.pixeltype not in {'float', 'double'}:
        scale = False # turn off scaling if image is discrete

    # handle `overlay` argument 
    if overlay is not None:
        if isinstance(overlay, str):
            overlay = iio2.image_read(overlay)
        if not isinstance(overlay, iio.ANTsImage):
            raise ValueError('overlay argument must be an ANTsImage')
        if overlay.dimension != 3:
            raise ValueError('Overlay image must have 3 dimensions!')

        if not iio.image_physical_space_consistency(image, overlay):
            overlay = reg.resample_image_to_target(overlay, image, interp_type='linear')

    # handle `slices` argument
    if slices is None:
        slices = [int(s/2) for s in image.shape]

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

    # handle xyz argument
    if xyz is not None:
        xyz_image = image.new_image_like(np.zeros(image.shape))
        xyz_image[xyz[0],:,:] = 1
        xyz_image[:,xyz[1],:] = 1
        xyz_image[:,:,xyz[2]] = 1

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
            vmin, vmax = image.quantile(scale)
        else:
            vmin = None
            vmax = None

        # resample image if spacing is very unbalanced
        spacing = [s for i,s in enumerate(image.spacing)]
        if (max(spacing) / min(spacing)) > 3.:
            new_spacing = (1,1,1)
            image = image.resample_image(tuple(new_spacing))
            if overlay is not None:
                overlay = overlay.resample_image(tuple(new_spacing))
            if xyz is not None:
                xyz_image = xyz_image.resample_image(tuple(new_spacing), interp_type=1)
            slices = [int(sl*(sold/snew)) for sl,sold,snew in zip(slices,spacing,new_spacing)]

        if not flat:
            nrow = 2
            ncol = 2
        else:
            nrow = 1
            ncol = 3

        fig = plt.figure(figsize=(10*figsize,10*figsize))

        gs = gridspec.GridSpec(nrow, ncol,
                 wspace=0.0, hspace=0.0, 
                 top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                 left=0.5/(ncol+1), right=1-0.5/(ncol+1))

        # pad image to have isotropic array dimensions
        old_shape = list(image.shape)
        max_shape = max(old_shape)
        new_shape = [max_shape]*3
        pad_vals = [(math.ceil((new-old)/2),math.floor((new-old)/2)) for new,old in zip(new_shape, old_shape)]
        image = np.pad(image.numpy(), pad_vals, mode='constant', constant_values=image.min())
        if overlay is not None:
            overlay = np.pad(overlay.numpy(), pad_vals, mode='constant', constant_values=image.min())
            overlay[np.abs(overlay) == 0] = np.nan
        if xyz is not None:
            xyz_image = np.pad(xyz_image.numpy(), pad_vals, mode='constant', constant_values=image.min())
            xyz_image[np.abs(xyz_image) == 0] = np.nan

        yz_slice = reorient_slice(image[slices[0],:,:],0)
        ax = plt.subplot(gs[0,1])
        ax.imshow(yz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            yz_overlay = reorient_slice(overlay[slices[0],:,:],0)
            ax.imshow(yz_overlay, alpha=overlay_alpha, cmap=overlay_cmap)
        if xyz is not None:
            yz_line = reorient_slice(xyz_image[slices[0],:,:],0)
            ax.imshow(yz_line, cmap=xyz_cmap, alpha=xyz_alpha)
        ax.axis('off')

        xz_slice = reorient_slice(image[:,slices[1],:],1)
        ax = plt.subplot(gs[0,0])
        ax.imshow(xz_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            xz_overlay = reorient_slice(overlay[:,slices[1],:],1)
            ax.imshow(xz_overlay, alpha=overlay_alpha, cmap=overlay_cmap)
        if xyz is not None:
            xz_line = reorient_slice(xyz_image[:,slices[1],:],1)
            ax.imshow(xz_line, cmap=xyz_cmap, alpha=xyz_alpha)
        ax.axis('off')

        xy_slice = reorient_slice(image[:,:,slices[2]],2)
        if not flat:
            ax = plt.subplot(gs[1,0])
        else:
            ax = plt.subplot(gs[0,2])
        ax.imshow(xy_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        if overlay is not None:
            xy_overlay = reorient_slice(overlay[:,:,slices[2]],2)
            ax.imshow(xy_overlay, alpha=overlay_alpha, cmap=overlay_cmap)
        if xyz is not None:
            xy_line = reorient_slice(xyz_image[:,:,slices[2]],2)
            ax.imshow(xy_line, cmap=xyz_cmap, alpha=xyz_alpha)
        ax.axis('off')

        if not flat:
            # empty corner
            ax = plt.subplot(gs[1,1])
            ax.imshow(np.zeros(image.shape[:-1]), cmap='Greys_r')
            ax.axis('off')

    ## multi-channel images ##
    elif image.components > 1:
        raise ValueError('Multi-channel images not currently supported!')

    if filename is not None:
        plt.savefig(filename, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    # turn warnings back to default
    warnings.simplefilter('default')


def plot(image, overlay=None, cmap='Greys_r', alpha=1, overlay_cmap='jet', overlay_alpha=0.9,
         axis=0, nslices=12, slices=None, ncol=4, slice_buffer=None, black_bg=True,
         bg_thresh_quant=0.01, bg_val_quant=0.99, domain_image_map=None, crop=False, scale=True,
         reverse=False, title=None, filename=None, dpi=500, figsize=1.5):
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
    if (axis == 'x') or (axis == 'saggittal'):
        axis = 0
    if (axis == 'y') or (axis == 'coronal'):
        axis = 1
    if (axis == 'z') or (axis == 'axial'):
        axis = 2

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
            vmin, vmax = image.quantile(scale)
        else:
            vmin = None
            vmax = None

        # Plot 2D image
        if image.dimension == 2:

            img_arr = image.numpy()
            img_arr = rotate90_matrix(img_arr)

            if not black_bg:
                img_arr[img_arr<image.quantile(bg_thresh_quant)] = image.quantile(bg_val_quant)

            if overlay is not None:
                ov_arr = overlay.numpy()
                ov_arr = rotate90_matrix(ov_arr)
                ov_arr[np.abs(ov_arr) == 0] = np.nan

            fig = plt.figure()
            ax = plt.subplot(111)

            # plot main image
            ax.imshow(img_arr, cmap=cmap,
                      alpha=alpha, 
                      vmin=vmin, vmax=vmax)

            if overlay is not None:

                ax.imshow(ov_arr, 
                          alpha=overlay_alpha, 
                          cmap=overlay_cmap)

            plt.axis('off')

        # Plot 3D image
        elif image.dimension == 3:
            # resample image if spacing is very unbalanced
            spacing = [s for i,s in enumerate(image.spacing) if i != axis]
            was_resampled = False
            if (max(spacing) / min(spacing)) > 3.:
                was_resampled = True
                new_spacing = (1,1,1)
                image = image.resample_image(tuple(new_spacing))
                if overlay is not None:
                    overlay = overlay.resample_image(tuple(new_spacing))

            img_arr = image.numpy()
            # reorder dims so that chosen axis is first
            img_arr = np.rollaxis(img_arr, axis)

            if overlay is not None:
                ov_arr = overlay.numpy()
                ov_arr[np.abs(ov_arr) == 0] = np.nan
                ov_arr = np.rollaxis(ov_arr, axis)

            if slices is None:
                if not isinstance(slice_buffer, (list, tuple)):
                    if slice_buffer is None:
                        slice_buffer = (int(img_arr.shape[1]*0.1), int(img_arr.shape[2]*0.1))
                    else:
                        slice_buffer = (slice_buffer, slice_buffer)
                nonzero = np.where(img_arr.sum(axis=(1,2)) > 0.01)[0]
                min_idx = nonzero[0] + slice_buffer[0]
                max_idx = nonzero[-1] - slice_buffer[1]
                slice_idxs = np.linspace(min_idx, max_idx, nslices).astype('int')
                if reverse:
                    slice_idxs = np.array(list(reversed(slice_idxs)))
            else:
                if isinstance(slices, (int,float)):
                    slices = [slices]
                # if all slices are less than 1, infer that they are relative slices
                if sum([s > 1 for s in slices]) == 0:
                    slices = [int(s*img_arr.shape[0]) for s in slices]
                slice_idxs = slices
                nslices = len(slices)

            if was_resampled:
                # re-calculate slices to account for new image shape
                slice_idxs = np.unique(np.array([int(s*(image.shape[axis]/img_arr.shape[0])) for s in slice_idxs]))

            # only have one row if nslices <= 6 and user didnt specify ncol
            if (nslices <= 6) and (ncol==4):
                ncol = nslices

            # calculate grid size
            nrow = math.ceil(nslices / ncol)
            xdim = img_arr.shape[2]
            ydim = img_arr.shape[1]

            dim_ratio = ydim/xdim
            fig = plt.figure(figsize=((ncol+1)*figsize*dim_ratio, (nrow+1)*figsize))

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
                        if not black_bg:
                        
                            imslice[imslice<image.quantile(bg_thresh_quant)] = image.quantile(bg_val_quant)
                    else:
                        imslice = np.zeros_like(img_arr[0])
                        imslice = reorient_slice(imslice, axis)

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
        plt.savefig(filename, dpi=dpi)
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







