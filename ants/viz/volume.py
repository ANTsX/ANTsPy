
__all__ = ['vol', 'vol_fold']

import os
import numpy as np
from tempfile import mktemp
import scipy.misc
import time
from tempfile import mktemp

import numpy as np
import scipy.misc

from .. import core
from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2


_view_map = {
    'left': (90,180,90),
    'inner_left': (90,180,90),
    'right': (90,0,270),
    'inner_right': (90,0,270),
    'front': (90,90,270),
    'back': (0,270,0),
    'top': (0,0,0),
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


def _vol_fold_single(image, outfile, magnification, dilation, inflation, alpha, overlay, overlay_mask, 
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
    #image_color = '%sx%.1f' % ('x'.join([str(grayscale)]*3), alpha)
    cmd = '-i [%s,0.0x1.0] ' % (image_tmp_file)

    # add mask
    #mask = image.clone() > 0.01
    #cm

    # display arg
    bg_grayscale = int(bg_grayscale*255)
    cmd += '-d %s[%.2f,%s,%s]' % (outfile,
                                  magnification,
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

        cmd += ' -f [%s,%s]' % (overlay_tmp_file, overlay_mask_tmp_file)

    if verbose:
        print(cmd)
        time.sleep(1)

    cmd = cmd.split(' ')
    libfn = utils.get_lib_fn('antsVol')
    retval = libfn(cmd)
    if retval != 0:
        print('ERROR: Non-Zero Return Value!')

    # cleanup temp file
    os.remove(image_tmp_file)
    if overlay is not None:
        os.remove(overlay_tmp_file)
        os.remove(overlay_mask_tmp_file)


def vol_fold(image, outfile,
    magnification=1.0, dilation=0, inflation=10, alpha=1.,
    overlay=None, overlay_mask=None, overlay_cmap='jet', overlay_scale=False, overlay_alpha=1.,
    rotation=None, cut_idx=None, cut_side='left', grayscale=0.7, bg_grayscale=0.9, 
    verbose=False, cleanup=True):
    """
    Generate a cortical folding volume of the gray matter of a brain image. 
    
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
    >>> ch2i = ants.image_read( ants.get_ants_data("mni") )
    >>> ch2seg = ants.threshold_image( ch2i, "Otsu", 3 )
    >>> wm   = ants.threshold_image( ch2seg, 2, 2 )
    >>> kimg = ants.weingarten_image_curvature( ch2i, 1.5  ).smooth_image( 1 )
    >>> rp = [(90,180,90), (90,180,270), (90,180,180)]
    >>> result = ants.vol_fold( wm, overlay=kimg, outfile='/users/ncullen/desktop/voltest.png')
    """
    # handle image arg
    if not isinstance(image, iio.ANTsImage):
        raise ValueError('image must be ANTsImage type')
    image = image.reorient_image2('RPI')

    # handle rotation arg
    if rotation is None:
        rotation = 'left'
    if not isinstance(rotation, list):
        rotation = [rotation]
    if not isinstance(rotation[0], list):
        rotation = [rotation]

    nrow = len(rotation)
    ncol = len(rotation[0])
    
    # handle outfile arg
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
                _vol_fold_single(image=image, outfile=ij_filename, magnification=magnification, 
                    dilation=dilation, inflation=inflation, alpha=alpha,
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



def convert_scalar_image_to_rgb(dimension, img, outimg, mask, colormap='red', custom_colormap_file=None, 
                                min_input=None, max_input=None, min_rgb_output=None, max_rgb_output=None,
                                vtk_lookup_table=None):
    """
    Usage: ConvertScalarImageToRGB imageDimension inputImage outputImage mask colormap [customColormapFile] [minimumInput] [maximumInput] [minimumRGBOutput=0] [maximumRGBOutput=255] <vtkLookupTable>
    Possible colormaps: grey, red, green, blue, copper, jet, hsv, spring, summer, autumn, winter, hot, cool, overunder, custom
    """
    if custom_colormap_file is None:
        custom_colormap_file = 'none'

    args = [dimension, img, outimg, mask, colormap, custom_colormap_file,
            min_input, max_input, min_rgb_output, max_rgb_output, vtk_lookup_table]
    processed_args = utils._int_antsProcessArguments(args)
    libfn = utils.get_lib_fn('ConvertScalarImageToRGB')
    libfn(processed_args)


def _vol_single(image, outfile, magnification, dilation, inflation, alpha, overlay, overlay_mask, 
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
    #image_color = '%sx%.1f' % ('x'.join([str(grayscale)]*3), alpha)
    cmd = '-i [%s,0.0x1.0] ' % (image_tmp_file)

    # add mask
    #mask = image.clone() > 0.01
    #cm

    # display arg
    bg_grayscale = int(bg_grayscale*255)
    cmd += '-d %s[%.2f,%s,%s]' % (outfile,
                                  magnification,
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

        cmd += ' -f [%s,%s]' % (overlay_tmp_file, overlay_mask_tmp_file)

    if verbose:
        print(cmd)
        time.sleep(1)

    cmd = cmd.split(' ')
    libfn = utils.get_lib_fn('antsVol')
    retval = libfn(cmd)
    if retval != 0:
        print('ERROR: Non-Zero Return Value!')

    # cleanup temp file
    os.remove(image_tmp_file)
    if overlay is not None:
        os.remove(overlay_tmp_file)
        os.remove(overlay_mask_tmp_file)


def vol(volume, overlays=None,
        quantlimits=(0.1,0.9),
        colormap='jet',
        rotation_params=(90,0,270),
        overlay_limits=None,
        magnification_factor=1.0,
        intensity_truncation=(0.0,1.0),
        filename=None,
        verbose=False):
    """
    Render an ANTsImage as a volume with optional ANTsImage functional overlay.
    This function is beautiful, and runs very fast. It requires VTK.
    
    ANTsR function: `antsrVol`
        NOTE: the ANTsPy version of this function does NOT make a function call
        to ANTs, unlike the ANTsR version, so you don't have to worry about paths.

    Arguments
    ---------
    volume : ANTsImage
        base volume to render
    
    overlay : list of ANTsImages
        functional overlay to render on the volume image. 
        These images should be in the same space

    colormap : string
        possible values:
            grey, red, green, blue, copper, jet, 
            hsv, spring, summer, autumn, winter, 
            hot, cool, overunder, custom

    rotation_params: tuple or collection of tuples or np.ndarray w/ shape (N,3)
        rotation parameters to render. The final image will be a stitch of each image
        from the given rotation params.
        e.g. if rotation_params = [(90,90,90),(180,180,180)], then the final
             stiched image will have 2 brain renderings at those angles

    overlay_limts

    magnification_factor : float
        how much to zoom in on the image before rendering. If the stitched images
        are too far apart, try increasing this value. If the brain volume gets
        cut off in the image, try decreasing this value

    intensity_truncation : 2-tuple of float
        percentile to truncate intensity of overlay

    filename : string
        final filename to which the final rendered volume stitch image will be saved
        this will always be a .png file

    verbose : boolean
        whether to print updates during rendering
    
    Returns
    -------
    - a numpy array representing the final stitched image.
    
    Effects
    -------
    - saves a few png files to disk

    Example
    -------
    >>> import ants
    >>> ch2i = ants.image_read( ants.get_ants_data("mni") )
    >>> ch2seg = ants.threshold_image( ch2i, "Otsu", 3 )
    >>> wm   = ants.threshold_image( ch2seg, 2, 2 )
    >>> kimg = ants.weingarten_image_curvature( ch2i, 1.5  ).smooth_image( 1 )
    >>> rp = [(90,180,90), (90,180,270), (90,180,180)]
    >>> result = ants.vol( wm, [kimg], quantlimits=(0.01,0.99), filename='/users/ncullen/desktop/voltest.png')
    """
    if (overlays is not None) and not isinstance(overlays, (list,iio.ANTsImage)):
        raise ValueError('overlay must be ANTsImage..')
    

    if not isinstance(colormap, list):
        colormap = [colormap]

    xfn = mktemp(suffix='.nii.gz')
    xmod = volume.clone()
    if (intensity_truncation[0] > 0) or (intensity_truncation[1] < 1):
        xmod = utils.iMath(volume, 'TruncateIntensity',
                           intensity_truncation[0], intensity_truncation[1])
    core.image_write(xmod, xfn)
    
    if filename is None:
        filename = mktemp()
    else:
        filename = os.path.expanduser(filename)
        if filename.endswith('.png'):
            filename = filename.replace('.png','')

    if not isinstance(rotation_params, np.ndarray):
        if isinstance(rotation_params, (tuple, list)):
            rotation_params = np.hstack(rotation_params)
        rotation_params = np.array(rotation_params)
    rotation_params = np.array(rotation_params).reshape(-1,3)

    pngs = []
    for myrot in range(rotation_params.shape[0]):
        volcmd = ['-i', xfn]
        if overlays is not None:
            if not isinstance(overlays, (tuple, list)):
                overlays = [overlays]
            ct = 0
            if len(colormap) != len(overlays):
                colormap = [colormap] * len(overlays)
            for overlay in overlays:
                ct = ct + 1
                wms = utils.smooth_image(overlay, 1.0)
                myquants = np.percentile(overlay[np.abs(overlay.numpy())>0], [q*100 for q in quantlimits])
                if overlay_limits is not None or (isinstance(overlay_limits, list) and (np.sum([o is not None for o in overlay_limits])>0)):
                    myquants = overlay_limits
                    overlay[overlay < myquants[0]] = 0
                    overlay[overlay > myquants[1]] = myquants[1]
                    if verbose: 
                        print(myquants)

                kblob = utils.threshold_image(wms, myquants[0], 1e15)
                kblobfn = mktemp(suffix='.nii.gz')
                core.image_write(kblob, kblobfn)
                overlayfn = mktemp(suffix='.nii.gz')
                core.image_write(overlay, overlayfn)

                csvlutfn = mktemp(suffix='.csv')
                overlayrgbfn = mktemp(suffix='.nii.gz')

                convert_scalar_image_to_rgb(dimension=3, img=overlayfn, outimg=overlayrgbfn, mask=kblobfn, colormap=colormap[ct-1],
                                    custom_colormap_file=None, min_input=myquants[0], max_input=myquants[1],
                                    min_rgb_output=0, max_rgb_output=255, vtk_lookup_table=csvlutfn)

                volcmd = volcmd + ['-f', ' [%s,%s]' % (overlayrgbfn, kblobfn)]
        
        if filename is None:
            volcmd = volcmd + [' -d [%s,%s]' % (magnification_factor, 'x'.join([str(r) for r in rotation_params[myrot,:]]))]            
        else:
            pngext = myrot
            if myrot < 10: pngext = '0%s' % pngext
            if myrot < 100: pngext = '0%s' % pngext
            pngfnloc = '%s%s.png' % (filename, pngext)
            try:
                os.remove(pngfnloc)
            except:
                pass
            rparamstring = 'x'.join([str(r) for r in rotation_params[myrot,:]])
            volcmd = volcmd + ['-d', '%s[%s,%s,255x255x255]' % (pngfnloc, magnification_factor, rparamstring)]

        ## C++ LIBRARY FUNCTION CALL ##
        libfn = utils.get_lib_fn('antsVol')
        retval = libfn(volcmd)

        if retval != 0:
            raise Exception('antsVol c++ function call failed for unknown reason')

        #if rotation_params.shape[0] > 1:
        pngs.append(pngfnloc)


    #if rotation_params.shape[0] > 1:
    mypngimg = scipy.misc.imread(pngs[0])
    img_shape = mypngimg.shape
    array_shape = (mypngimg.shape[0], mypngimg.shape[1]*len(pngs), mypngimg.shape[-1])
    mypngarray = np.zeros(array_shape).astype('uint8')
    for i in range(len(pngs)):
        mypngimg = scipy.misc.imread(pngs[i])
        mypngarray[:,(i*img_shape[1]):((i+1)*img_shape[1]),:] = mypngimg

    scipy.misc.imsave('%s.png' % filename, mypngarray)

    return mypngarray























