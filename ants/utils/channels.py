
 

__all__ = ['merge_channels',
           'split_channels']

from ..core import ants_image as iio
from .. import lib


_supported_ptypes = {'unsigned char', 'unsigned int', 'float', 'double'}
_short_ptype_map = {
    'unsigned char' : 'UC',
    'unsigned int': 'UI',
    'float': 'F',
    'double' : 'D'
}

# pick up lib.mergeChannelsX functions
_merge_channels_dict = {}
for ndim in {2,3,4}:
    _merge_channels_dict[ndim] = {}
    for d1 in _supported_ptypes:
        d1a = _short_ptype_map[d1]
        _merge_channels_dict[ndim][d1] = 'mergeChannels%s%i'%(d1a,ndim)

# pick up lib.splitChannelsX functions
_split_channels_dict = {}
for ndim in {2,3,4}:
    _split_channels_dict[ndim] = {}
    for d1 in _supported_ptypes:
        d1a = _short_ptype_map[d1]
        _split_channels_dict[ndim][d1] = 'splitChannels%s%i'%(d1a,ndim)


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

    merge_channels_fn = lib.__dict__[_merge_channels_dict[img_list[0].pixeltype][img_list[0].dimension]]
    img = merge_channels_fn([img.pointer for img in img_list])
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
    split_channels_fn = lib.__dict__[_split_channels_dict[img.pixeltype][img.dimension]]
    itkimgs = split_channels_fn(img.pointer)
    antsimgs = [iio.ANTsImage(itkimg) for itkimg in itkimgs]
    return antsimgs




