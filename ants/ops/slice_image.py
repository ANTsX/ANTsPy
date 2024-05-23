
__all__ = ['slice_image']

import math
import numpy as np
import ants
from ants.decorators import image_method
from ants.internal import get_lib_fn

@image_method
def slice_image(image, axis, idx, collapse_strategy=0):
    """
    Slice an image.

    Arguments
    ---------
    axis: integer 
        Which axis.

    idx: integer
        Which slice number.    

    collapse_strategy:  integer
        Collapse strategy for sub-matrix: 0, 1, or 2.  0: collapse to sub-matrix 
        if positive-definite.  Otherwise throw an exception. Default.  1: Collapse 
        to identity.  2:  Collapse to sub-matrix if positive definite. Otherwise
        collapse to identity.

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> mni2 = ants.slice_image(mni, axis=1, idx=100)
    """
    if image.has_components:
        ilist = ants.split_channels(image)
        if image.dimension == 2:
            return np.stack(tuple([i.slice_image(axis, idx, collapse_strategy) for i in ilist]), axis=-1)
        else:
            return ants.merge_channels([i.slice_image(axis, idx, collapse_strategy) for i in ilist])
    
    if axis == -1:
        axis = image.dimension - 1
        
    if axis > (image.dimension - 1) or axis < 0:
        raise Exception('The axis must be between 0 and image.dimension - 1')
        
    if image.dimension == 2:
        if axis == 0:
            return image[idx,:]
        elif axis == 1:
            return image[:,idx]
        raise Exception('Parameters not understood for 2D image.')
        
    if collapse_strategy != 0 and collapse_strategy != 1 and collapse_strategy != 2:
        raise ValueError('collapse_strategy must be 0, 1, or 2.') 

    inpixeltype = image.pixeltype
    ndim = image.dimension
    if image.pixeltype != 'float':
        image = image.clone('float')

    libfn = get_lib_fn('sliceImage')
    itkimage = libfn(image.pointer, axis, idx, collapse_strategy)

    return ants.from_pointer(itkimage).clone(inpixeltype)


