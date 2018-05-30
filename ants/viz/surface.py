
__all__ = ['surf', 'surf_fold', 'surf_smooth', 'get_canonical_views']

import os
import time
from tempfile import mktemp

import numpy as np
import scipy.misc

from .. import core
from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2


_view_map = {
    'left': (270,0,270),
    'inner_left': (270,0,270),
    'right': (270,0,90),
    'inner_right': (270,0,90),
    'front': (270,0,0),
    'back': (270,0,180),
    'top': (0,0,180),
    'bottom':(180,0,0)
}


def get_canonical_views():
    """
    Get the canonical views used for surface and volume rendering. You can use
    this as a reference for slightly altering rotation parameters in ants.surf
    and ants.vol functions.

    Note that these views are for images that have 'RPI' orientation. 
    Images are automatically reoriented to RPI in ANTs surface and volume rendering
    functions but you can reorient images yourself with `img.reorient_image2('RPI')
    """
    return _view_map


def _surf_fold_single(image, outfile, dilation, inflation, alpha, overlay, overlay_mask, 
    overlay_cmap, overlay_scale, overlay_alpha, rotation, 
    cut_idx, cut_side, grayscale, bg_grayscale, verbose):
    """
    Helper function for making a single surface fold image.
    """
    if rotation is None:
        rotation = (270,0,270)
    if not isinstance(rotation, (str, tuple)):
        raise ValueError('rotation must be a tuple or string')
    if isinstance(rotation, tuple):
        if isinstance(rotation[0], str):
            rotation_dx = rotation[1]
            rotation = rotation[0]
            if 'inner' in rotation:
                if rotation.count('_') == 2:
                    rsplit = rotation.split('_')
                    rotation = '_'.join(rsplit[:-1])
                    cut_idx = int(rsplit[-1])
                else:
                    cut_idx = 0
                centroid = int(-1*image.origin[0] + image.get_center_of_mass()[0])
                cut_idx = centroid + cut_idx
                cut_side = rotation.replace('inner_','')
            else:
                cut_idx = int(image.get_centroids()[0][0])
            rotation_string = rotation
            rotation = _view_map[rotation.lower()]
            rotation = (r+rd for r,rd in zip(rotation,rotation_dx))

    elif isinstance(rotation, str):
        if 'inner' in rotation:
            if rotation.count('_') == 2:
                rsplit = rotation.split('_')
                rotation = '_'.join(rsplit[:-1])
                cut_idx = int(rsplit[-1])
            else:
                cut_idx = 0
            centroid = int(-1*image.origin[0] + image.get_center_of_mass()[0])
            if verbose:
                print('Found centroid at %i index' % centroid)
            cut_idx = centroid + cut_idx
            cut_side = rotation.replace('inner_','')
            if verbose:
                print('Cutting image on %s side at %i index' % (cut_side,cut_idx))
        else:
            cut_idx = int(image.get_centroids()[0][0])
        rotation_string = rotation
        rotation = _view_map[rotation.lower()]

    # handle filename argument
    outfile = os.path.expanduser(outfile)

    # handle overlay argument
    if overlay is not None:
        if not iio.image_physical_space_consistency(image, overlay):
            overlay = overlay.resample_image_to_target(image)
            if verbose:
                print('Resampled overlay to base image space')

        if overlay_mask is None:
            overlay_mask = image.iMath_MD(3)

    ## PROCESSING ##
    if dilation > 0:
        image = image.iMath_MD(dilation)
        
    thal = image
    wm = image
    #wm = wm + thal
    wm = wm.iMath_fill_holes().iMath_get_largest_component().iMath_MD()
    wms = wm.smooth_image(0.5)
    wmt_label = wms.iMath_propagate_labels_through_mask(thal, 500, 0 )
    image = wmt_label.threshold_image(1,1)
    if cut_idx is not None:
        if cut_idx > image.shape[0]:
            raise ValueError('cut_idx (%i) must be less than image X dimension (%i)' % (cut_idx, image.shape[0]))
        cut_mask = image*0 + 1.
        if 'inner' in rotation_string:
            if cut_side == 'left':
                cut_mask[cut_idx:,:,:] = 0
            elif cut_side == 'right':
                cut_mask[:cut_idx,:,:] = 0
            else:
                raise ValueError('cut_side argument must be `left` or `right`')
        else:
            if 'left' in rotation:
                cut_mask[cut_idx:,:,:] = 0
            elif 'right' in rotation:
                cut_mask[:cut_idx,:,:] = 0
        image = image * cut_mask
    ##

    # surface arg
    # save base image to temp file
    image_tmp_file = mktemp(suffix='.nii.gz')
    image.to_file(image_tmp_file)
    # build image color
    grayscale = int(grayscale*255)
    image_color = '%sx%.1f' % ('x'.join([str(grayscale)]*3), alpha)
    cmd = '-s [%s,%s] ' % (image_tmp_file, image_color)

    # anti-alias arg
    tolerance = 0.01
    cmd += '-a %.3f ' % tolerance

    # inflation arg
    cmd += '-i %i ' % inflation

    # display arg
    bg_grayscale = int(bg_grayscale*255)
    cmd += '-d %s[%s,%s]' % (outfile,
                              'x'.join([str(s) for s in rotation]),
                              'x'.join([str(bg_grayscale)]*3))

    # overlay arg
    if overlay is not None:
        #-f [rgbImageFileName,maskImageFileName,<alpha=1>]
        if overlay_scale == True:
            min_overlay, max_overlay = overlay.quantile((0.05,0.95))
            overlay[overlay<min_overlay] = min_overlay
            overlay[overlay>max_overlay] = max_overlay
        elif isinstance(overlay_scale, tuple):
            min_overlay, max_overlay = overlay.quantile((overlay_scale[0], overlay_scale[1]))
            overlay[overlay<min_overlay] = min_overlay
            overlay[overlay>max_overlay] = max_overlay

        # make tempfile for overlay
        overlay_tmp_file = mktemp(suffix='.nii.gz')
        # convert overlay image to RGB
        overlay.scalar_to_rgb(mask=overlay_mask, cmap=overlay_cmap,
                              filename=overlay_tmp_file)
        # make tempfile for overlay mask
        overlay_mask_tmp_file = mktemp(suffix='.nii.gz')
        overlay_mask.to_file(overlay_mask_tmp_file)

        cmd += ' -f [%s,%s,%.2f]' % (overlay_tmp_file, overlay_mask_tmp_file, overlay_alpha)

    if verbose:
        print(cmd)
        time.sleep(1)

    cmd = cmd.split(' ')
    libfn = utils.get_lib_fn('antsSurf')
    retval = libfn(cmd)
    if retval != 0:
        print('ERROR: Non-Zero Return Value!')

    # cleanup temp file
    os.remove(image_tmp_file)
    if overlay is not None:
        os.remove(overlay_tmp_file)
        os.remove(overlay_mask_tmp_file)


def surf_fold(image, outfile,
    # processing args
    dilation=0, inflation=10, alpha=1.,
    # overlay args
    overlay=None, overlay_mask=None, overlay_cmap='jet', overlay_scale=False,
    overlay_alpha=1.,
    # display args
    rotation=None, cut_idx=None, cut_side='left',
    grayscale=0.7, bg_grayscale=0.9,
    verbose=False, cleanup=True):
    """
    Generate a cortical folding surface of the gray matter of a brain image. 
    
    rotation : 3-tuple | string | 2-tuple of string & 3-tuple
        if 3-tuple, this will be the rotation from RPI about x-y-z axis
        if string, this should be a canonical view (see : ants.get_canonical_views())
        if 2-tuple, the first value should be a string canonical view, and the second
                    value should be a 3-tuple representing a delta change in each
                    axis from the canonical view (useful for apply slight changes
                    to canonical views)
        NOTE:
            rotation=(0,0,0) will be a view of the top of the brain with the 
                       front of the brain facing the bottom of the image
        NOTE:
            1st value : controls rotation about x axis (anterior/posterior tilt)
                note : the x axis extends to the side of you
            2nd value : controls rotation about y axis (inferior/superior tilt)
                note : the y axis extends in front and behind you
            3rd value : controls rotation about z axis (left/right tilt) 
                note : thte z axis extends up and down

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> seg = mni.otsu_segmentation(k=3)
    >>> wm_img = seg.threshold_image(3,3)
    >>> ants.surf_fold(wm_img, outfile='~/desktop/surf_fold.png')
    >>> # with overlay
    >>> overlay = ants.weingarten_image_curvature( mni, 1.5  ).smooth_image( 1 )
    >>> ants.surf_fold(image=wm_img, overlay=overlay, outfile='~/desktop/surf_fold2.png')
    """
    if not isinstance(rotation, list):
        rotation = [rotation]
    if not isinstance(rotation[0], list):
        rotation = [rotation]

    nrow = len(rotation)
    ncol = len(rotation[0])

    #image = image.reorient_image2('RPI')
    #if overlay is not None:
    #    overlay = overlay.reorient_image2('RPI')
    
    # preprocess outfile arg
    outfile = os.path.expanduser(outfile)
    if not outfile.endswith('.png'):
        outfile = outfile.split('.')[0] + '.png'

    # create all of the individual filenames by appending to outfile
    rotation_filenames = []
    for rowidx in range(nrow):
        rotation_filenames.append([])
        for colidx in range(ncol):
            if rotation[rowidx][colidx] is not None:
                ij_filename = outfile.replace('.png','_%i%i.png' % (rowidx,colidx))
            else:
                ij_filename = None
            rotation_filenames[rowidx].append(ij_filename)

    # create each individual surface image
    for rowidx in range(nrow):
        for colidx in range(ncol):
            ij_filename = rotation_filenames[rowidx][colidx]
            if ij_filename is not None:
                ij_rotation = rotation[rowidx][colidx]
                _surf_fold_single(image=image, outfile=ij_filename, dilation=dilation, inflation=inflation, alpha=alpha,
                                  overlay=overlay, overlay_mask=overlay_mask, overlay_cmap=overlay_cmap, 
                                  overlay_scale=overlay_scale,overlay_alpha=overlay_alpha,rotation=ij_rotation, 
                                  cut_idx=cut_idx,cut_side=cut_side,grayscale=grayscale, 
                                  bg_grayscale=bg_grayscale,verbose=verbose)
            rotation_filenames[rowidx][colidx] = ij_filename

    # if only one view just rename the file, otherwise stitch images together according
    # to the `rotation` list structure
    if (nrow==1) and (ncol==1):
        os.rename(rotation_filenames[0][0], outfile)
    else:
        if verbose:
            print('Stitching images together..')
        # read first image to calculate shape of stitched image
        first_actual_file = None
        for rowidx in range(nrow):
            for colidx in range(ncol):
                if rotation_filenames[rowidx][colidx] is not None:
                    first_actual_file = rotation_filenames[rowidx][colidx]
                    break

        if first_actual_file is None:
            raise ValueError('No images were created... check your rotation argument')

        mypngimg = scipy.misc.imread(first_actual_file)
        img_shape = mypngimg.shape
        array_shape = (mypngimg.shape[0]*nrow, mypngimg.shape[1]*ncol, mypngimg.shape[-1])
        mypngarray = np.zeros(array_shape).astype('uint8')

        # read each individual image and place it in the larger stitch
        for rowidx in range(nrow):
            for colidx in range(ncol):
                ij_filename = rotation_filenames[rowidx][colidx]
                if ij_filename is not None:
                    mypngimg = scipy.misc.imread(ij_filename)
                else:
                    mypngimg = np.zeros(img_shape) + int(255*bg_grayscale)
                
                row_start = rowidx*img_shape[0]
                row_end   = (rowidx+1)*img_shape[0]
                col_start = colidx*img_shape[1]
                col_end   = (colidx+1)*img_shape[1]
                
                mypngarray[row_start:row_end,col_start:col_end:] = mypngimg

        # save the stitch to the outfile
        scipy.misc.imsave(outfile, mypngarray)

        # delete all of the individual images if cleanup arg is True
        if cleanup:
            for rowidx in range(nrow):
                for colidx in range(ncol):
                    ij_filename = rotation_filenames[rowidx][colidx]
                    if ij_filename is not None:
                        os.remove(ij_filename)


def _surf_smooth_single(image,outfile,dilation,smooth,threshold,inflation,alpha,
    cut_idx,cut_side,overlay,overlay_mask,overlay_cmap,overlay_scale, 
    overlay_alpha,rotation,grayscale,bg_grayscale,verbose):
    """
    Generate a surface of the smooth white matter of a brain image. 

    This is great for displaying functional activations as are typically seen
    in the neuroimaging literature.

    Arguments
    ---------
    image : ANTsImage
        A binary segmentation of the white matter surface.
        If you don't have a white matter segmentation, you can use
        `kmeans_segmentation` or `atropos` on a full-brain image.
    
    inflation : integer
        how much to inflate the final surface

    rotation : 3-tuple | string | list of 3-tuples | list of string
        if tuple, this is rotation of X, Y, Z 
        if string, this is a canonical view..
            Options: 'left', 'right', 'inner_left', 'inner_right', 
            'anterior', 'posterior', 'inferior', 'superior'
        if list of tuples or strings, the surface images will be arranged
        in a grid according to the shape of the list.
        
        e.g. rotation=[['left', 'inner_left' ],
                       ['right','inner_right']] 
        will result in a 2x2 grid of the above 4 canonical views
    
    grayscale : float
        value between 0 and 1 representing how light to make the base image.
        grayscale = 1 will make the base image completely white and 
        grayscale = 0 will make the base image completely black

    background : float
        value between 0 and 1 representing how light to make the base image.
        see `grayscale` arg.

    outfile : string
        filepath to which the surface plot will be saved

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> seg = mni.otsu_segmentation(k=3)
    >>> wm_img = seg.threshold_image(3,3)
    >>> #ants.surf_smooth(wm_img, outfile='~/desktop/surf_smooth.png')
    >>> ants.surf_smooth(wm_img, rotation='inner_right', outfile='~/desktop/surf_smooth_innerright.png')
    >>> # with overlay
    >>> overlay = ants.weingarten_image_curvature( mni, 1.5  ).smooth_image( 1 ).iMath_GD(3)
    >>> ants.surf_smooth(image=wm_img, overlay=overlay, outfile='~/desktop/surf_smooth2.png')
    """

    # handle rotation argument
    if rotation is None:
        rotation = (270,0,270)
    if not isinstance(rotation, (str, tuple)):
        raise ValueError('rotation must be a 3-tuple or string')
    if isinstance(rotation, str):
        if 'inner' in rotation:
            cut_idx = int(image.shape[2]/2)
            cut_side = rotation.replace('inner_','')
        rotation = _view_map[rotation.lower()]


    # handle filename argument
    if outfile is None:
        outfile = mktemp(suffix='.png')
    else:
        outfile = os.path.expanduser(outfile)

    # handle overlay argument
    if overlay is not None:
        if overlay_mask is None:
            overlay_mask = image.iMath_MD(3)

    # PROCESSING IMAGE
    image = image.reorient_image2('RPI')
    image = image.iMath_fill_holes().iMath_get_largest_component()
    if dilation > 0:
        image = image.iMath_MD(dilation)
    if smooth > 0:
        image = image.smooth_image(smooth)
    if threshold > 0:
        image = image.threshold_image(threshold)
    if cut_idx is not None:
        if cut_side == 'left':
            image = image.crop_indices((0,0,0),(cut_idx,image.shape[1],image.shape[2]))
        elif cut_side == 'right':
            image = image.crop_indices((cut_idx,0,0),image.shape)
        else:
            raise ValueError('not valid cut_side argument')

    # surface arg
    # save base image to temp file
    image_tmp_file = mktemp(suffix='.nii.gz')
    image.to_file(image_tmp_file)
    # build image color
    grayscale = int(grayscale*255)
    alpha = 1.
    image_color = '%sx%.1f' % ('x'.join([str(grayscale)]*3),
                               alpha)
    cmd = '-s [%s,%s] ' % (image_tmp_file, image_color)

    # anti-alias arg
    tolerance = 0.01
    cmd += '-a %.3f ' % tolerance

    # inflation arg
    cmd += '-i %i ' % inflation

    # display arg
    bg_grayscale = int(bg_grayscale*255)
    cmd += '-d %s[%s,%s]' % (outfile,
                              'x'.join([str(s) for s in rotation]),
                              'x'.join([str(bg_grayscale)]*3))

    # overlay arg
    if overlay is not None:
        overlay = overlay.reorient_image2('RPI')
        if overlay_scale == True:
            min_overlay, max_overlay = overlay.quantile((0.05,0.95))
            overlay[overlay<min_overlay] = min_overlay
            overlay[overlay>max_overlay] = max_overlay
        elif isinstance(overlay_scale, tuple):
            min_overlay, max_overlay = overlay.quantile((overlay_scale[0], overlay_scale[1]))
            overlay[overlay<min_overlay] = min_overlay
            overlay[overlay>max_overlay] = max_overlay

        # make tempfile for overlay
        overlay_tmp_file = mktemp(suffix='.nii.gz')
        # convert overlay image to RGB
        overlay.scalar_to_rgb(mask=overlay_mask, cmap=overlay_cmap,
                              filename=overlay_tmp_file)
        # make tempfile for overlay mask
        overlay_mask_tmp_file = mktemp(suffix='.nii.gz')
        overlay_mask.to_file(overlay_mask_tmp_file)

        cmd += ' -f [%s,%s,%.2f]' % (overlay_tmp_file, overlay_mask_tmp_file, overlay_alpha)

    if verbose:
        print(cmd)
        time.sleep(1)

    cmd = cmd.split(' ')
    libfn = utils.get_lib_fn('antsSurf')
    retval = libfn(cmd)
    if retval != 0:
        print('ERROR: Non-Zero Return Value!')

    # cleanup temp file
    os.remove(image_tmp_file)


def surf_smooth(image, outfile,
    # processing args
    dilation=1.0, smooth=1.0, threshold=0.5, inflation=200, alpha=1.,
    cut_idx=None, cut_side='left',
    # overlay args
    overlay=None, overlay_mask=None, overlay_cmap='jet', overlay_scale=False, 
    overlay_alpha=1.,
    # display args
    rotation=None,
    grayscale=0.7, bg_grayscale=0.9,
    # extraneous args
    verbose=False, cleanup=True):
    """
    Generate a cortical folding surface of the gray matter of a brain image. 
    
    rotation : 3-tuple | string | 2-tuple of string & 3-tuple
        if 3-tuple, this will be the rotation from RPI about x-y-z axis
        if string, this should be a canonical view (see : ants.get_canonical_views())
        if 2-tuple, the first value should be a string canonical view, and the second
                    value should be a 3-tuple representing a delta change in each
                    axis from the canonical view (useful for apply slight changes
                    to canonical views)
        NOTE:
            rotation=(0,0,0) will be a view of the top of the brain with the 
                       front of the brain facing the bottom of the image
        NOTE:
            1st value : controls rotation about x axis (anterior/posterior tilt)
                note : the x axis extends to the side of you
            2nd value : controls rotation about y axis (inferior/superior tilt)
                note : the y axis extends in front and behind you
            3rd value : controls rotation about z axis (left/right tilt) 
                note : thte z axis extends up and down

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> seg = mni.otsu_segmentation(k=3)
    >>> wm_img = seg.threshold_image(3,3)
    >>> ants.surf_fold(wm_img, outfile='~/desktop/surf_fold.png')
    >>> # with overlay
    >>> overlay = ants.weingarten_image_curvature( mni, 1.5  ).smooth_image( 1 )
    >>> ants.surf_fold(image=wm_img, overlay=overlay, outfile='~/desktop/surf_fold2.png')
    """
    if not isinstance(rotation, list):
        rotation = [rotation]
    if not isinstance(rotation[0], list):
        rotation = [rotation]

    nrow = len(rotation)
    ncol = len(rotation[0])
    
    # preprocess outfile arg
    outfile = os.path.expanduser(outfile)
    if not outfile.endswith('.png'):
        outfile = outfile.split('.')[0] + '.png'

    # create all of the individual filenames by appending to outfile
    rotation_filenames = []
    for rowidx in range(nrow):
        rotation_filenames.append([])
        for colidx in range(ncol):
            if rotation[rowidx][colidx] is not None:
                ij_filename = outfile.replace('.png','_%i%i.png' % (rowidx,colidx))
            else:
                ij_filename = None
            rotation_filenames[rowidx].append(ij_filename)

    # create each individual surface image
    for rowidx in range(nrow):
        for colidx in range(ncol):
            ij_filename = rotation_filenames[rowidx][colidx]
            if ij_filename is not None:
                ij_rotation = rotation[rowidx][colidx]
                _surf_smooth_single(image=image,outfile=ij_filename,rotation=ij_rotation,
                                    dilation=dilation,smooth=smooth,threshold=threshold,
                                    inflation=inflation,alpha=alpha,cut_idx=cut_idx,
                                    cut_side=cut_side,overlay=overlay,overlay_mask=overlay_mask,
                                    overlay_cmap=overlay_cmap,overlay_scale=overlay_scale,
                                    overlay_alpha=overlay_alpha,grayscale=grayscale,
                                    bg_grayscale=bg_grayscale,verbose=verbose)
            rotation_filenames[rowidx][colidx] = ij_filename

    # if only one view just rename the file, otherwise stitch images together according
    # to the `rotation` list structure
    if (nrow==1) and (ncol==1):
        os.rename(rotation_filenames[0][0], outfile)
    else:
        # read first image to calculate shape of stitched image
        first_actual_file = None
        for rowidx in range(nrow):
            for colidx in range(ncol):
                if rotation_filenames[rowidx][colidx] is not None:
                    first_actual_file = rotation_filenames[rowidx][colidx]
                    break

        if first_actual_file is None:
            raise ValueError('No images were created... check your rotation argument')

        mypngimg = scipy.misc.imread(first_actual_file)
        img_shape = mypngimg.shape
        array_shape = (mypngimg.shape[0]*nrow, mypngimg.shape[1]*ncol, mypngimg.shape[-1])
        mypngarray = np.zeros(array_shape).astype('uint8')

        # read each individual image and place it in the larger stitch
        for rowidx in range(nrow):
            for colidx in range(ncol):
                ij_filename = rotation_filenames[rowidx][colidx]
                if ij_filename is not None:
                    mypngimg = scipy.misc.imread(ij_filename)
                else:
                    mypngimg = np.zeros(img_shape) + int(255*bg_grayscale)
                
                row_start = rowidx*img_shape[0]
                row_end   = (rowidx+1)*img_shape[0]
                col_start = colidx*img_shape[1]
                col_end   = (colidx+1)*img_shape[1]
                
                mypngarray[row_start:row_end,col_start:col_end:] = mypngimg

        # save the stitch to the outfile
        scipy.misc.imsave(outfile, mypngarray)

        # delete all of the individual images if cleanup arg is True
        if cleanup:
            for rowidx in range(nrow):
                for colidx in range(ncol):
                    ij_filename = rotation_filenames[rowidx][colidx]
                    if ij_filename is not None:
                        os.remove(ij_filename)



def surf(x, y=None, z=None,
         quantlimits=(0.1,0.9),
         colormap='jet',
         grayscale=0.7,
         bg_grayscale=0.9,
         alpha=None,
         inflation_factor=0,
         tol=0.03,
         smoothing_sigma=0.0,
         rotation_params=(90,0,270),
         overlay_limits=None,
         filename=None,
         verbose=False):
    """
    Render a function onto a surface.

    ANTsR function: `antsrSurf`
        NOTE: the ANTsPy version of this function does NOT make a function call
        to ANTs, unlike the ANTsR version, so you don't have to worry about paths.

    Arguments
    ---------
    x : ANTsImage   
        input image defining the surface on which to render
    
    y : ANTsImage
        input image list defining the function to render on the surface. 
        these image(s) should be in the same space as x.
    
    z : ANTsImage
        input image list mask for each y function to render on the surface. 
        these image(s) should be in the same space as y.
    
    quantlimits : tuple/list
        lower and upper quantile limits for overlay
    
    colormap : string
        one of: grey, red, green, blue, copper, jet, hsv, spring, summer, 
        autumn, winter, hot, cool, overunder, custom
    
    alpha : scalar  
        transparency vector for underlay and each overlay, default zero
    
    inflation_factor : integer
        number of inflation iterations to run

    tol : float
        error tolerance for surface reconstruction. Smaller values will
        lead to better surfaces, at the cost of taking longer.
        Try decreasing this value if your surfaces look very block-y.
    
    smoothing_sigma : scalar
        gaussian smooth the overlay by this sigma
    
    rotation_params : tuple/list/ndarray 
        3 Rotation angles expressed in degrees or a matrix of rotation 
        parameters that will be applied in sequence.
    
    overlay_limits : tuple (optional)
        absolute lower and upper limits for functional overlay. this parameter 
        will override quantlimits. Currently, this will set levels above 
        overlayLimits[2] to overlayLimits[1]. Can be a list of length of y.
    
    filename : string
        prefix filename for output pngs
    
    verbose : boolean
        prints the command used to call antsSurf
    
    Returns
    -------
    N/A

    Example
    -------
    >>> import ants
    >>> ch2i = ants.image_read( ants.get_ants_data("ch2") )
    >>> ch2seg = ants.threshold_image( ch2i, "Otsu", 3 )
    >>> wm   = ants.threshold_image( ch2seg, 3, 3 )
    >>> wm2 = wm.smooth_image( 1 ).threshold_image( 0.5, 1e15 )
    >>> kimg = ants.weingarten_image_curvature( ch2i, 1.5  ).smooth_image( 1 )
    >>> wmz = wm2.iMath("MD",3)
    >>> rp = [(90,180,90), (90,180,270), (90,180,180)]
    >>> ants.surf( x=wm2, y=[kimg], z=[wmz],
                inflation_factor=255, overlay_limits=(-0.3,0.3), verbose = True,
                rotation_params = rp, filename='/users/ncullen/desktop/surface.png')

    """
    TEMPFILES = []
    len_x = len(x) if isinstance(x, (tuple,list)) else 1
    len_y = len(y) if isinstance(y, (tuple,list)) else 1
    len_z = len(z) if isinstance(z, (tuple,list)) else 1

    if alpha is None:
        alpha = [1] * (len_x+len_y)

    if len_z != len_y:
        raise ValueError('each y must have a mask in z')

    if (overlay_limits is not None) and not isinstance(overlay_limits, (tuple, list)):
        overlay_limits = [overlay_limits]

    # not supported right now
    domain_image_map = None
    if domain_image_map is not None:
        pass

    if filename is None:
        filename = mktemp()
        #TEMPFILES.append(filename)
    else:
        filename = os.path.expanduser(filename)
        if filename.endswith('.png'):
            filename = filename.replace('.png','')

    if not isinstance(rotation_params, np.ndarray):
        if isinstance(rotation_params, (tuple, list)):
            rotation_params = np.hstack(rotation_params)
        rotation_params = np.array(rotation_params)
    rotation_params = np.array(rotation_params).reshape(-1,3)

    if (not isinstance(y, (tuple,list))) and (y is not None):
        y = [y]
    if (not isinstance(z, (tuple,list))) and (z is not None):
        z = [z]

    xfn = mktemp(suffix='.nii.gz')
    TEMPFILES.append(xfn)
    core.image_write(x, xfn)

    pngs = []
    gs = int(grayscale*255)
    background_color = '%ix%ix%ix%s' % (gs,gs,gs,str(alpha[0]))

    for myrot in range(rotation_params.shape[0]):
        surfcmd = ['-s', '[%s,%s]' %(xfn,background_color)]

        if y is not None:
            ct = 0
            if len(colormap) != len(y):
                colormap = [colormap] * len(y)

            for overlay in y:
                ct = ct + 1
                wms = utils.smooth_image(overlay, smoothing_sigma)
                myquants = np.percentile(wms[np.abs(wms.numpy())>0], [q*100 for q in quantlimits])

                if overlay_limits is not None or (isinstance(overlay_limits, list) and \
                    (np.sum([o is not None for o in overlay_limits])>0)):
                    myquants = overlay_limits

                kblobfn = mktemp(suffix='.nii.gz')
                TEMPFILES.append(kblobfn)
                core.image_write(z[ct-1], kblobfn)
                overlayfn = mktemp(suffix='.nii.gz')
                TEMPFILES.append(overlayfn)
                core.image_write(wms, overlayfn)
                csvlutfn = mktemp(suffix='.csv')
                TEMPFILES.append(csvlutfn)
                overlayrgbfn = mktemp(suffix='.nii.gz')
                TEMPFILES.append(overlayrgbfn)
                iio.scalar_to_rgb(dimension=3, img=overlayfn, outimg=overlayrgbfn, 
                    mask=kblobfn, colormap=colormap[ct-1], custom_colormap_file=None, 
                    min_input=myquants[0], max_input=myquants[1],
                    min_rgb_output=0, max_rgb_output=255, vtk_lookup_table=csvlutfn)
                alphaloc = alpha[min(ct, len(alpha)-1)]

                surfcmd = surfcmd + ['-f', '[%s,%s,%s]' % (overlayrgbfn, kblobfn,str(alphaloc))]

        rparamstring = 'x'.join([str(rp) for rp in rotation_params[myrot,:]])
        pngext = myrot
        if myrot < 10:
            pngext = '0%s' % pngext
        if myrot < 100:
            pngext = '0%s' % pngext

        pngfnloc = '%s%s.png' % (filename, pngext)
        try:
            os.remove(pngfnloc)
        except:
            pass

        gs2 = int(bg_grayscale * 255.)
        surfcmd += ['-d', '%s[%s,%ix%ix%i]'%(pngfnloc,rparamstring,gs2,gs2,gs2)]
        surfcmd += ['-a', '%f' % tol]
        surfcmd += ['-i', '%i' % inflation_factor]

        libfn = utils.get_lib_fn('antsSurf')
        libfn(surfcmd)

        if rotation_params.shape[0] > 1:
            pngs.append(pngfnloc)

    # CLEANUP TEMP FILES
    for tfile in TEMPFILES:
        try:
            os.remove(tfile)
        except:
            pass

