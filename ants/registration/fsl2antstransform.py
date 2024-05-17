

__all__ = ['fsl2antstransform']

from .. import utils
from ..core import ants_transform as tio

def fsl2antstransform(matrix, reference, moving):
    """
    Convert an FSL linear transform to an antsrTransform
    
    ANTsR function: `fsl2antsrtransform`

    Arguments
    ---------
    matrix : ndarray/list
        4x4 matrix of transform parameters

    reference : ANTsImage
        target image

    moving : ANTsImage
        moving image

    Returns
    -------
    ANTsTransform

    Examples
    --------
    >>> import ants
    >>> import numpy as np
    >>> fslmat = np.zeros((4,4))
    >>> np.fill_diagonal(fslmat, 1)
    >>> img = ants.image_read(ants.get_ants_data('ch2'))
    >>> tx = ants.fsl2antstransform(fslmat, img, img)
    """
    if reference.dimension != 3:
        raise ValueError('reference image must be 3 dimensions')

    if reference.pixeltype != 'float':
        reference = reference.clone('float')
    if moving.pixeltype != 'float':
        moving = moving.clone('float')

    libfn = utils.get_lib_fn('fsl2antstransformF3')
    tx_ptr = libfn(list(matrix), 
                    reference.pointer,
                    moving.pointer,
                    1)

    return tio.ANTsTransform(precision='float', dimension=reference.dimension, 
                             transform_type='AffineTransform', pointer=tx_ptr)

