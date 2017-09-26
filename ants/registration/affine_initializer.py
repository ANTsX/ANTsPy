
__all__ = ['affine_initializer']

import warnings
from tempfile import mktemp

from .. import utils

def affine_initializer(fixed_image, moving_image, search_factor=20,
                        radian_fraction=0.1, use_principal_axis=False, 
                        local_search_iterations=10, mask=None, txfn=None ):
    """
    A multi-start optimizer for affine registration
    Searches over the sphere to find a good initialization for further
    registration refinement, if needed.  This is a arapper for the ANTs
    function antsAffineInitializer.
    
    ANTsR function: `affineInitializer`

    Arguments
    ---------
    fixed_image : ANTsImage
        the fixed reference image
    moving_image : ANTsImage 
        the moving image to be mapped to the fixed space
    search_factor : scalar
        degree of increments on the sphere to search
    radian_fraction : scalar
        between zero and one, defines the arc to search over
    use_principal_axis : boolean
        boolean to initialize by principal axis
    local_search_iterations : scalar
        gradient descent iterations
    mask : ANTsImage (optional)
        optional mask to restrict registration
    txfn : string (optional)
        filename for the transformation

    Returns
    -------
    ndarray
        transformation matrix
    
    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16'))
    >>> mi = ants.image_read(ants.get_ants_data('r27'))
    >>> txfile = ants.affine_initializer( fi, mi )
    >>> tx = ants.read_transform(txfile, dimension=2)
    """

    if txfn is None:
        txfn = mktemp(suffix='.mat')

    veccer = [fixed_image.dimension, fixed_image, moving_image, txfn,
                search_factor, radian_fraction, int(use_principal_axis),
                local_search_iterations]
    if mask is not None:
        veccer.append(mask)

    xxx = utils._int_antsProcessArguments(veccer)
    libfn = utils.get_lib_fn('antsAffineInitializer')
    retval = libfn(xxx)

    if retval != 0:
        warnings.warn('ERROR: Non-zero exit status!')
    
    return txfn

