
 

__all__ = ['reorient_image',
           'get_center_of_mass']

import os
import numpy as np
from tempfile import mktemp

from . import apply_transforms
from .. import utils


def reorient_image(image, axis1, axis2=None, doreflection=False, doscale=0, txfn=None):
    """
    Align image along a specified axis

    ANTsR function: `reorientImage`
    
    Arguments
    ---------
    image : ANTsImage
        image to reorient
    
    axis1 : list/tuple of integers
        vector of size dim, might need to play w/axis sign
    
    axis2 : list/tuple of integers
        vector of size dim for 3D
    
    doreflection : boolean
        whether to reflect
    
    doscale : scalar value
         1 allows automated estimate of scaling
    
    txfn : string
        file name for transformation
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> ants.reorient_image(fi, (1,0))
    """
    inpixeltype = image.pixeltype
    if image.pixeltype != 'float':
        image = image.clone('float')

    axis_was_none = False
    if axis2 is None:
        axis_was_none = True
        axis2 = [0]*image.dimension

    axis1 = np.array(axis1)
    axis2 = np.array(axis2)

    axis1 = axis1 / np.sqrt(np.sum(axis1*axis1)) * (-1)
    axis1 = axis1.astype('int')

    if not axis_was_none:
        axis2 = axis2 / np.sqrt(np.sum(axis2*axis2)) * (-1)
        axis2 = axis2.astype('int')
    else:
        axis2 = np.array([0]*image.dimension).astype('int')

    if txfn is None:
        txfn = mktemp(suffix='.mat')

    if isinstance(doreflection, tuple):
        doreflection = list(doreflection)
    if not isinstance(doreflection, list):
        doreflection = [doreflection]

    if isinstance(doscale, tuple):
        doscale = list(doscale)
    if not isinstance(doscale, list):
        doscale = [doscale]

    if len(doreflection) == 1:
        doreflection = [doreflection[0]]*image.dimension
    if len(doscale) == 1:
        doscale = [doscale[0]]*image.dimension

    libfn = utils.get_lib_fn('reorientImage%s%i' % image._libsuffix)
    libfn(image.pointer, txfn, axis1.tolist(), axis2.tolist(), doreflection, doscale)
    image2 = apply_transforms(image, image, transformlist=[txfn])

    if image.pixeltype != inpixeltype:
        image2 = image2.clone(inpixeltype)

    return {'reoimage':image2,
            'txfn':txfn}


def get_center_of_mass(image):
    """
    Compute an image center of mass in physical space which is defined 
    as the mean of the intensity weighted voxel coordinate system.

    ANTsR function: `getCenterOfMass`
    
    Arguments
    ---------
    image : ANTsImage
        image from which center of mass will be computed

    Returns
    -------
    scalar

    Example
    -------
    >>> fi = ants.image_read( ants.get_ants_data("r16"))
    >>> com1 = ants.get_center_of_mass( fi )
    >>> fi = ants.image_read( ants.get_ants_data("r64"))
    >>> com2 = ants.get_center_of_mass( fi )
    """
    if image.pixeltype != 'float':
        image = image.clone('float')

    libfn = utils.get_lib_fn('centerOfMass%s%i' % image._libsuffix)
    com = libfn(image.pointer)

    return tuple(com)



