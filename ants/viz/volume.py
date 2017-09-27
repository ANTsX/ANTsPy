
__all__ = ['vol']

import os
import numpy as np
from tempfile import mktemp
import scipy.misc

from .. import core
from ..core import ants_image as iio
from .. import utils


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
    >>> wm   = ants.threshold_image( ch2seg, 3, 3 )
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























