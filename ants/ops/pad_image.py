
__all__ = ['pad_image']

import math

import ants
from ants.decorators import image_method
from ants.internal import get_lib_fn

@image_method
def pad_image(image, shape=None, pad_width=None, value=0.0, return_padvals=False):
    """
    Pad an image to have the given shape or to be isotropic.

    Arguments
    ---------
    image : ANTsImage
        image to pad

    shape : tuple
        - if shape is given, the image will be padded in each dimension
          until it has this shape
        - if shape and pad_width are both None, the image will be padded along
          each dimension to match the largest existing dimension so that it has
          isotropic dimensions.

    pad_width : list of integers or list-of-list of integers
        How much to pad in each direction. If a single list is
        supplied (e.g., [4,4,4]), then the image will be padded by half
        that amount on both sides. If a list-of-list is supplied
        (e.g., [(0,4),(0,4),(0,4)]), then the image will be
        padded unevenly on the different sides

    pad_value : scalar
        value with which image will be padded

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> img2 = ants.pad_image(img, shape=(300,300))
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> mni2 = ants.pad_image(mni)
    >>> mni3 = ants.pad_image(mni, pad_width=[(0,4),(0,4),(0,4)])
    >>> mni4 = ants.pad_image(mni, pad_width=(4,4,4))
    """
    inpixeltype = image.pixeltype
    ndim = image.dimension
    if image.pixeltype != 'float':
        image = image.clone('float')

    if pad_width is None:
        if shape is None:
            shape = [max(image.shape)] * image.dimension
        lower_pad_vals = [math.floor(max(ns-os,0)/2) for os,ns in zip(image.shape, shape)]
        upper_pad_vals = [math.ceil(max(ns-os,0)/2) for os,ns in zip(image.shape, shape)]
    else:
        if shape is not None:
            raise ValueError('Cannot give both `shape` and `pad_width`. Pick one!')
        if len(pad_width) != image.dimension:
            raise ValueError('Must give pad width for each image dimension')

        lower_pad_vals = []
        upper_pad_vals = []
        for p in pad_width:
            if isinstance(p, (list, tuple)):
                lower_pad_vals.append(p[0])
                upper_pad_vals.append(p[1])
            else:
                lower_pad_vals.append(math.floor(p/2))
                upper_pad_vals.append(math.ceil(p/2))

    libfn = get_lib_fn('padImage')
    itkimage = libfn(image.pointer, lower_pad_vals, upper_pad_vals, value)

    new_image = ants.from_pointer(itkimage).clone(inpixeltype)
    if return_padvals:
        return new_image, lower_pad_vals, upper_pad_vals
    else:
        return new_image


