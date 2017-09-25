
__all__ = ['surf']

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


def surf(x, y=None, z=None,
         quantlimits=(0.1,0.9),
         colormap='jet',
         alpha=None,
         inflation_factor=25,
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
    >>> wm2 = ants.image_read('~/desktop/wm2.nii.gz')
    >>> kimg2 = ants.image_read('~/desktop/kimg2.nii.gz')
    >>> wmz = ants.image_read('~/desktop/wmz.nii.gz')
    >>> rps = [(90,180,90), (90,180,270), (90,180,180)]
    >>> ants.Surf( x=wm2, y=[kimg], z=[wmz],
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
    background_color = '255x255x255x%s' % (str(alpha[0]))

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
                convert_scalar_image_to_rgb(dimension=3, img=overlayfn, outimg=overlayrgbfn, 
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

        surfcmd = surfcmd + ['-d', '%s[%s,255x255x255]'%(pngfnloc,rparamstring)]
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

