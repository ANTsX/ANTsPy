
__all__ = ['slice_image']

import math

from ..core import ants_image as iio
from .. import utils


def slice_image(image, axis=None, idx=None):
    """
    Slice an image.

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> mni2 = ants.slice_image(mni, axis=1, idx=100)
    """
    if image.dimension < 3:
        raise ValueError('image must have at least 3 dimensions')

    inpixeltype = image.pixeltype
    ndim = image.dimension
    if image.pixeltype != 'float':
        image = image.clone('float')

    libfn = utils.get_lib_fn('sliceImageF%i' % ndim)
    itkimage = libfn(image.pointer, axis, idx)

    return iio.ANTsImage(pixeltype='float', dimension=ndim-1, 
                         components=image.components, pointer=itkimage).clone(inpixeltype)


