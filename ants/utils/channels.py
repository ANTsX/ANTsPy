
 

__all__ = ['merge_channels',
           'split_channels']

from ..core import ants_image as iio
from .. import lib


_merge_channels_dict = {
    'unsigned char' : {
        2: lib.mergeChannelsUC2,
        3: lib.mergeChannelsUC3,
        4: lib.mergeChannelsUC4
    },
    'unsigned int' : {
        2: lib.mergeChannelsUI2,
        3: lib.mergeChannelsUI3,
        4: lib.mergeChannelsUI4
    },
    'float' : {
        2: lib.mergeChannelsF2,
        3: lib.mergeChannelsF3,
        4: lib.mergeChannelsF4
    },
    'double' : {
        2: lib.mergeChannelsD2,
        3: lib.mergeChannelsD3,
        4: lib.mergeChannelsD4
    }
}


_split_channels_dict = {
    'unsigned char' : {
        2: lib.splitChannelsUC2,
        3: lib.splitChannelsUC3,
        4: lib.splitChannelsUC4
    },
    'unsigned int' : {
        2: lib.splitChannelsUI2,
        3: lib.splitChannelsUI3,
        4: lib.splitChannelsUI4
    },
    'float' : {
        2: lib.splitChannelsF2,
        3: lib.splitChannelsF3,
        4: lib.splitChannelsF4
    },
    'double' : {
        2: lib.splitChannelsD2,
        3: lib.splitChannelsD3,
        4: lib.splitChannelsD4
    }
}


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

    merge_channels_fn = _merge_channels_dict[img_list[0].pixeltype][img_list[0].dimension]
    img = merge_channels_fn([img._img for img in img_list])
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
    split_channels_fn = _split_channels_dict[img.pixeltype][img.dimension]
    itkimgs = split_channels_fn(img._img)
    antsimgs = [iio.ANTsImage(itkimg) for itkimg in itkimgs]
    return antsimgs




