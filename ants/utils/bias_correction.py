
 

__all__ = ['n3_bias_field_correction',
           'n4_bias_field_correction']


from . import process_args as pargs

from ..core import ants_image as iio
from .get_mask import get_mask
from .. import lib


def n3_bias_field_correction(img, downsample_factor=3):
    """
    N3 Bias Field Correction

    ANTsR function: `n3BiasFieldCorrection`

    Arguments
    ---------
    img : ANTsImage
        image to be bias corrected

    downsample_factor : scalar
        how much to downsample image before performing bias correction

    Returns
    -------
    ANTsImage
    
    Example
    -------
    >>> img = ants.image_read( ants.get_ants_data('r16') )
    >>> img_n3 = ants.n3_bias_field_correction(img)
    """
    outimg = img.clone()
    args = [img.dimension, img, outimg, downsample_factor]
    processed_args = pargs._int_antsProcessArguments(args)
    lib.N3BiasFieldCorrection(processed_args)
    return outimg


def n4_bias_field_correction(img, mask=None, shrink_factor=4,
                             convergence={'iters':[50,50,50,50], 'tol':1e-07},
                             spline_param=200, verbose=False, weight_mask=None):
    """
    N4 Bias Field Correction

    ANTsR function: `n4BiasFieldCorrection`

    Arguments
    ---------
    img : ANTsImage
        image to bias correct
    
    mask : ANTsImage   
        input mask, if one is not passed one will be made
    
    shrink_factor : scalar   
        Shrink factor for multi-resolution correction, typically integer less than 4
    
    convergence : dict w/ keys `iters` and `tol`
        iters : vector of maximum number of iterations for each shrinkage factor
        tol : the convergence tolerance.
    
    spline_param : integer
        Parameter controlling number of control points in spline. Either single value, indicating how many control points, or vector with one entry per dimension of image, indicating the spacing in each direction.
    
    verbose : boolean
        enables verbose output.
    
    weight_mask : ANTsImage (optional)
        antsImage of weight mask

    Returns
    -------
    ANTsImage
    
    Example
    -------
    >>> img = ants.image_read( ants.get_ants_data('r16') )
    >>> img_n4 = ants.n4_bias_field_correction(img)
    """
    iters = convergence['iters']
    tol = convergence['tol']
    if mask is None:
        mask = get_mask(img)

    N4_CONVERGENCE_1 = '[%s, %.10f]' % ('x'.join([str(it) for it in iters]), tol)
    N4_SHRINK_FACTOR_1 = str(shrink_factor)
    if (not isinstance(spline_param, (tuple,list))) or (len(spline_param) == 1):
        N4_BSPLINE_PARAMS = '[%i]' % spline_param
    elif (isinstance(spline_param)) and (len(spline_param) == img.dimension):
        N4_BSPLINE_PARAMS = '[%s]' % ('x'.join([str(sp) for sp in spline_param]))
    else:
        raise ValueError('Length of splineParam must either be 1 or dimensionality of image')

    if weight_mask is not None:
        if not isinstance(weight_mask, iio.ANTsImage):
            raise ValueError('Weight Image must be an antsImage')

    outimg = img.clone()
    kwargs = {
        'd': outimg.dimension,
        'i': img,
        'w': weight_mask,
        's': N4_SHRINK_FACTOR_1,
        'c': N4_CONVERGENCE_1,
        'b': N4_BSPLINE_PARAMS,
        'x': mask,
        'o': outimg,
        'v': int(verbose)
    }

    processed_args = pargs._int_antsProcessArguments(kwargs)
    lib.N4BiasFieldCorrection(processed_args)
    return outimg


