
 

__all__ = ['merge_channels',
           'split_channels']

from ..core import ants_image as iio
from .. import utils


def merge_channels(img_list):
    """
    Merge channels of multiple scalar ANTsImage types into one 
    multi-channel ANTsImage
    
    ANTsR function: `mergeChannels`

    Arguments
    ---------
    img_list : list/tuple of ANTsImage types
        scalar images to merge
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> img2 = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> img3 = ants.merge_channels([img,img2])
    >>> img3.components == 2
    """
    inpixeltype = img_list[0].pixeltype
    for img in img_list:
        if not isinstance(img, iio.ANTsImage):
            raise ValueError('list may only contain ANTsImage objects')
        if img.pixeltype != inpixeltype:
            raise ValueError('all images must have the same pixeltype')

    libfn = utils.get_lib_fn('mergeChannels%s%i'%(utils.short_type(img_list[0].pixeltype),img_list[0].dimension))
    img = libfn([img.pointer for img in img_list])
    return iio.ANTsImage(img)


def split_channels(img):
    """
    Split channels of a multi-channel ANTsImage into a collection
    of scalar ANTsImage types
    
    Arguments
    ---------
    img : ANTsImage
        multi-channel image to split

    Returns
    -------
    list of ANTsImage types

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> img2 = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> imgmerge = ants.merge_channels([img,img2])
    >>> imgmerge.components == 2
    >>> imgs_unmerged = ants.split_channels(imgmerge)
    >>> len(imgs_unmerged) == 2
    >>> imgs_unmerged[0].components == 1
    """
    libfn = utils.get_lib_fn('splitChannels%s%i' % (utils.short_type(img.pixeltype), img.dimension))
    itkimgs = libfn(img.pointer)
    antsimgs = [iio.ANTsImage(itkimg) for itkimg in itkimgs]
    return antsimgs




