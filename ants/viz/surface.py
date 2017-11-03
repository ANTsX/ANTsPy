
__all__ = ['surf', 'surf_fold', 'surf_smooth', 'get_canonical_views']

import os
import numpy as np
import time
from tempfile import mktemp

from .. import core
from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2


_view_map = {
    'left': (270,0,270),
    'inner_left': (270,0,90),
    'right': (270,0,90),
    'inner_right': (270,0,270)
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


def surf_fold(wm, outfile, 
            # processing args
            inflation=10, 
            # overlay args
            #overlay=None, overlay_mask=None, overlay_dilation=1., overlay_smooth=1.,
            # display args
            rotation=None, grayscale=0.7, bg_grayscale=0.9,
            verbose=False):
    """
    Generate a cortical folding surface of the gray matter of a brain image. 

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> seg = mni.otsu_segmentation(k=3)
    >>> wm_img = seg.threshold_image(3,3)
    >>> ants.surf_fold(wm_img, outfile='~/desktop/surf_fold_example.png')
    """
    # handle rotation argument
    if rotation is None:
        rotation = (270,0,270)
    if not isinstance(rotation, (str, tuple)):
        raise ValueError('rotation must be a tuple or string')
    if isinstance(rotation, str):
        rotation = _view_map[rotation.lower()]

    # handle filename argument
    if outfile is None:
        outfile = mktemp(suffix='.png')
    else:
        outfile = os.path.expanduser(outfile)

    ## PROCESSING ##
    thal = wm
    #wm = wm + thal
    wm = wm.iMath_fill_holes().iMath_get_largest_component().iMath_MD()
    wms = wm.smooth_image(0.5)
    wmt_label = wms.iMath_propagate_labels_through_mask(thal, 500, 0 )
    image = wmt_label.threshold_image(1,1)
    ##

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


def surf_smooth_multi(image, outfile,
                      # processing args
                      dilation=1.0, smooth=1.0, threshold=0.5, inflation=200, 
                      # overlay args
                      #overlay=None, overlay_mask=None, overlay_dilation=1., overlay_smooth=1.,
                      # display args
                      rotation=[['left','inner_left'],
                                ['right','inner_right']], 
                      grayscale=0.7, bg_grayscale=0.9,
                      # extraneous args
                      verbose=False):
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
    >>> ants.surf_smooth(wm_img, outfile='~/desktop/surf_smooth_example.png')
    """
    # handle rotation argument
    if rotation is None:
        rotation = (270,0,270)
    if not isinstance(rotation, (str, tuple)):
        raise ValueError('rotation must be a 3-tuple or string')
    if isinstance(rotation, str):
        rotation = _view_map[rotation.lower()]

    # handle filename argument
    if outfile is None:
        outfile = mktemp(suffix='.png')
    else:
        outfile = os.path.expanduser(outfile)

    # preprocessing white matter segmentation
    image = image.reorient_image2('RPI')
    image = image.iMath_fill_holes().iMath_get_largest_component()
    if dilation > 0:
        image = image.iMath_MD(dilation)
    if smooth > 0:
        image = image.smooth_image(smooth)
    if threshold > 0:
        image = image.threshold_image(threshold)

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
            dilation=1.0, smooth=1.0, threshold=0.5, inflation=200, 
            # overlay args
            #overlay=None, overlay_mask=None, overlay_dilation=1., overlay_smooth=1.,
            # display args
            rotation=[['left','inner_left'],['right','inner_right']], 
            grayscale=0.7, bg_grayscale=0.9,
            # extraneous args
            verbose=False):
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
    >>> ants.surf_smooth(wm_img, outfile='~/desktop/surf_smooth_example.png')
    """
    # handle rotation argument
    if rotation is None:
        rotation = (270,0,270)
    if not isinstance(rotation, (str, tuple)):
        raise ValueError('rotation must be a 3-tuple or string')
    if isinstance(rotation, str):
        rotation = _view_map[rotation.lower()]

    # handle filename argument
    if outfile is None:
        outfile = mktemp(suffix='.png')
    else:
        outfile = os.path.expanduser(outfile)

    # preprocessing white matter segmentation
    image = image.reorient_image2('RPI')
    image = image.iMath_fill_holes().iMath_get_largest_component()
    if dilation > 0:
        image = image.iMath_MD(dilation)
    if smooth > 0:
        image = image.smooth_image(smooth)
    if threshold > 0:
        image = image.threshold_image(threshold)

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

