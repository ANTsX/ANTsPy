
__all__ = ['surf']

import os
import numpy as np
from tempfile import mktemp

from .. import core
from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2


def surf2(image, 
          #overlay=None, overlay_mask=None, overlay_colormap='jet',
          smooth=0.5, dilation=1., threshold=0.5, scale_intensity=True,
          inflation=50,
          views=None, rotation=None):
    """
    views : string or list of strings
        Canonical views of the surface.
        Options: left, right, inner_left, inner_right, 
                 inferior, superior, anterior, posterios
    
COMMAND: 
     antsSurf

OPTIONS: 
     -s, --surface-image surfaceImageFilename
                         [surfaceImageFilename,<defaultColor=255x255x255x1>]
     -m, --mesh meshFilename
     -f, --functional-overlay [rgbImageFileName,maskImageFileName,<alpha=1>]
     -a, --anti-alias-rmse value
     -i, --inflation numberOfIterations
                     [numberOfIterations]
     -d, --display doWindowDisplay
                   filename
                   <filename>[rotateXxrotateYxrotateZ,<backgroundColor=255x255x255>]
     -o, --output surfaceFilename
                  imageFilename[spacing]
     -b, --scalar-bar lookupTable
                      [lookupTable,<title=antsSurf>,<numberOfLabels=5>,<widthxheight>]
     -h 
     --help 

    """
    view_map = {
        'left': (270,0,270),
        'inner_left': (270,0,90),
        'right': (270,0,90),
        'inner_right': (270,0,270)
    }
    image = image.threshold_image(image.min()+0.01)
    image = image.iMath_MD(dilation).reorient_image('RPI')
    image = image.smooth_image(smooth).threshold_image(0.5)

    cmd = '-s %s -m %s -f %s -a %s -i %s -d %s -o %s -b %s'




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

