
 

__all__ = ['n3_bias_field_correction',
           'n4_bias_field_correction',
           'abp_n4']


from . import process_args as pargs
from .get_mask import get_mask
from .iMath import iMath

from ..core import ants_image as iio
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


def abp_n4(img, intensity_truncation=(0.025,0.975,256), mask=None, usen3=False):
    """
    Truncate outlier intensities and bias correct with the N4 algorithm.

    Arguments
    ---------
    img : ANTsImage
        image to correct and truncate

    intensity_truncation : 3-tuple
        quantiles for intensity truncation

    mask : ANTsImage (optional)
        mask for bias correction

    usen3 : boolean
        if True, use N3 bias correction instead of N4

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> img2 = ants.abp_n4(img)
    """
    if len(intensity_truncation) != 3:
        raise ValueError('intensity_truncation must have 3 values')

    outimg = iMath(img, 'TruncateIntensity', 
            intensity_truncation[0], intensity_truncation[1], intensity_truncation[2])
    if usen3 == True:
        outimg = n3_bias_field_correction(outimg, 4)
        outimg = n3_bias_field_correction(outimg, 2)
        return outimg
    else:
        outimg = n4_bias_field_correction(outimg, mask)
        return outimg


