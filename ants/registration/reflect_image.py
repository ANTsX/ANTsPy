 

__all__ = ['reflect_image']

from tempfile import mktemp

from ..core import ants_image as iio
from ..core import ants_image_io as iio2

from .. import utils
from .. import lib

from .interface import registration
from .apply_transforms import apply_transforms


_supported_ptypes = {'unsigned char', 'unsigned int', 'float', 'double'}
_short_ptype_map = {'unsigned char' : 'UC',
                    'unsigned int': 'UI',
                    'float': 'F',
                    'double' : 'D'}

# pick up lib.reflectionMatrix functions
_reflection_matrix_dict = {}
for ndim in {2,3}:
    _reflection_matrix_dict[ndim] = {}
    for d1 in _supported_ptypes:
        d1a = _short_ptype_map[d1]
        _reflection_matrix_dict[ndim][d1] = 'reflectionMatrix%s%i'%(d1a,ndim)


def reflect_image(img, axis=None, tx=None, metric='mattes'):
    """
    Reflect an image along an axis

    ANTsR function: `reflectImage`

    Arguments
    ---------
    img : ANTsImage
        image to reflect
    
    axis : integer (optional)
        which dimension to reflect across, numbered from 0 to imageDimension-1
    
    tx : string (optional)
        transformation type to estimate after reflection
    
    metric : string  
        similarity metric for image registration. see antsRegistration.
    
    Returns
    -------
    ANTsImage

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

    reflection_matrix_fn = lib.__dict__[_reflection_matrix_dict[img.pixeltype][img.dimension]]
    reflection_matrix_fn(img.pointer, axis, rflct)

    if tx is not None:
        rfi = registration(img, img, type_of_transform=tx,
                            syn_metric=metric, outprefix=mktemp(),
                            initial_transform=rflct)
        return rfi
    else:
        return apply_transforms(img, img, rflct)

