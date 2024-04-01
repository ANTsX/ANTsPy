"""
Apply anti-alias filter on a binary ANTsImage
"""

__all__ = ['anti_alias']

from ..core import ants_image as iio
from ..core import ants_image_io as iio2
from .. import utils


def anti_alias(image):
    """
    Apply Anti-Alias filter to a binary image
    
    ANTsR function: N/A

    Arguments
    ---------
    image : ANTsImage
        binary image to which anti-aliasing will be applied

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> mask = ants.get_mask(img)
    >>> mask_aa = ants.anti_alias(mask)
    >>> ants.plot(mask)
    >>> ants.plot(mask_aa)
    """
    if image.pixeltype != 'unsigned char':
        if image.max() > 255.:
            image = (image - image.max()) / (image.max() - image.min())
        image = image.clone('unsigned char')

    libfn = utils.get_lib_fn('antiAlias%s' % image._libsuffix)
    new_ptr = libfn(image.pointer)
    return iio.ANTsImage(pixeltype='float', dimension=image.dimension, 
                         components=image.components, pointer=new_ptr)
