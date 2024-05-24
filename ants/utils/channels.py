
 

__all__ = ['merge_channels',
           'split_channels']




import ants
from ants.internal import get_lib_fn
from ants.decorators import image_method


def merge_channels(image_list, channels_first=False):
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
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> image2 = ants.image_read(ants.get_ants_data('r16'))
    >>> image3 = ants.merge_channels([image,image2])
    >>> image3 = ants.merge_channels([image,image2], channels_first=True)
    >>> image3.numpy()
    >>> image3.components == 2
    """
    inpixeltype = image_list[0].pixeltype
    dimension = image_list[0].dimension
    components = len(image_list)

    for image in image_list:
        if not ants.is_image(image):
            raise ValueError('list may only contain ANTsImage objects')
        if image.pixeltype != inpixeltype:
            raise ValueError('all images must have the same pixeltype')

    libfn = get_lib_fn('mergeChannels')
    image_ptr = libfn([image.pointer for image in image_list])
    
    image = ants.from_pointer(image_ptr)
    image.channels_first = channels_first
    return image

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

    libfn = get_lib_fn('splitChannels')
    itkimages = libfn(image.pointer)
    antsimages = [ants.from_pointer(itkimage) for itkimage in itkimages]
    return antsimages




