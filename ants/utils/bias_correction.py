
 

__all__ = ['n3_bias_field_correction',
           'n4_bias_field_correction']


from . import process_args as pargs

from ..core import ants_image as iio
from .get_mask import get_mask
from .. import lib


def n3_bias_field_correction(img, downsample_factor=3):
    """
    N3 Bias Field Correction
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


