 

__all__ = ['reflect_image']

from tempfile import mktemp

from ..core import ants_image as iio
from ..core import image_io as iio2

from .. import utils
from .. import lib

from .interface import registration
from .apply_transforms import apply_transforms


_reflection_matrix_dict = {
    'unsigned char': {
        2: lib.reflectionMatrixUC2,
        3: lib.reflectionMatrixUC3
    },
    'unsigned int': {
        2: lib.reflectionMatrixUI2,
        3: lib.reflectionMatrixUI3
    },
    'float': {
        2: lib.reflectionMatrixF2,
        3: lib.reflectionMatrixF3
    },
    'double': {
        2: lib.reflectionMatrixD2,
        3: lib.reflectionMatrixD3
    }
}

def reflect_image(img, axis=None, tx=None, metric='mattes'):
    """
    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16'), 'float' )
    >>> axis = 2
    >>> asym = ants.reflect_image(fi, axis, 'Affine')['warpedmovout']
    >>> asym = asym - fi
    """
    if axis is None:
        axis = img.dimension - 1

    if (axis > img.dimension) or (axis < 0):
        axis = img.dimension - 1

    rflct = mktemp(suffix='.mat')

    reflection_matrix_fn = _reflection_matrix_dict[img.pixeltype][img.dimension]
    reflection_matrix_fn(img._img, axis, rflct)

    if tx is not None:
        rfi = registration(img, img, type_of_transform=tx,
                            syn_metric=metric, outprefix=mktemp(),
                            initial_transform=rflct)
        return rfi
    else:
        return apply_transforms(img, img, rflct)

