
__all__ = ['slice_image']

import math

from ..core import ants_image as iio
from .. import utils


def slice_image(image, axis=None, idx=None, collapse_strategy=0):
    """
    Slice an image.

    Arguments
    ---------
    axis: integer 
        Which axis.

    idx: integer
        Which slice number.    

    collapse_strategy:  integer
        Collapse strategy for sub-matrix: 0, 1, or 2.  0: collapse to sub-matrix 
        if positive-definite.  Otherwise throw an exception. Default.  1: Collapse 
        to identity.  2:  Collapse to sub-matrix if positive definite. Otherwise
        collapse to identity.

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> mni2 = ants.slice_image(mni, axis=1, idx=100)
    """
    if image.dimension < 3:
        raise ValueError('image must have at least 3 dimensions')

    if collapse_strategy != 0 and collapse_strategy != 1 and collapse_strategy != 2:
        raise ValueError('collapse_strategy must be 0, 1, or 2.') 

    inpixeltype = image.pixeltype
    ndim = image.dimension
    if image.pixeltype != 'float':
        image = image.clone('float')

    libfn = utils.get_lib_fn('sliceImageF%i' % ndim)
    itkimage = libfn(image.pointer, axis, idx, collapse_strategy)

    return iio.ANTsImage(pixeltype='float', dimension=ndim-1, 
                         components=image.components, pointer=itkimage).clone(inpixeltype)


