
 

__all__ = ['merge_channels',
           'split_channels']

from ants.decorators import image_method
from ..core import ants_image_io as iio2
from ..core import ants_image as iio
from .. import utils


def merge_channels(image_list):
    """
    Merge channels of multiple scalar ANTsImage types into one 
    multi-channel ANTsImage
    
    ANTsR function: `mergeChannels`

    Arguments
    ---------
    image_list : list/tuple of ANTsImage types
        scalar images to merge
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> image2 = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> image3 = ants.merge_channels([image,image2])
    >>> image3.components == 2
    """
    inpixeltype = image_list[0].pixeltype
    dimension = image_list[0].dimension
    components = len(image_list)

    for image in image_list:
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('list may only contain ANTsImage objects')
        if image.pixeltype != inpixeltype:
            raise ValueError('all images must have the same pixeltype')

    libfn = utils.get_lib_fn('mergeChannels')
    image_ptr = libfn([image.pointer for image in image_list])
    
    return iio2.from_pointer(image_ptr)

@image_method
def split_channels(image):
    """
    Split channels of a multi-channel ANTsImage into a collection
    of scalar ANTsImage types
    
    Arguments
    ---------
    image : ANTsImage
        multi-channel image to split

    Returns
    -------
    list of ANTsImage types

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> image2 = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> imagemerge = ants.merge_channels([image,image2])
    >>> imagemerge.components == 2
    >>> images_unmerged = ants.split_channels(imagemerge)
    >>> len(images_unmerged) == 2
    >>> images_unmerged[0].components == 1
    """
    inpixeltype = image.pixeltype
    dimension = image.dimension
    components = 1

    libfn = utils.get_lib_fn('splitChannels')
    itkimages = libfn(image.pointer)
    antsimages = [iio2.from_pointer(itkimage) for itkimage in itkimages]
    return antsimages




