 

__all__ = ['reflect_image']

from tempfile import mktemp

from .. import utils
from .interface import registration
from .apply_transforms import apply_transforms


def reflect_image(image, axis=None, tx=None, metric='mattes'):
    """
    Reflect an image along an axis

    ANTsR function: `reflectImage`

    Arguments
    ---------
    image : ANTsImage
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
        axis = image.dimension - 1

    if (axis > image.dimension) or (axis < 0):
        axis = image.dimension - 1

    rflct = mktemp(suffix='.mat')

    libfn = utils.get_lib_fn('reflectionMatrix%s'%image._libsuffix)
    libfn(image.pointer, axis, rflct)

    if tx is not None:
        rfi = registration(image, image, type_of_transform=tx,
                            syn_metric=metric, outprefix=mktemp(),
                            initial_transform=rflct)
        return rfi
    else:
        return apply_transforms(image, image, rflct)

