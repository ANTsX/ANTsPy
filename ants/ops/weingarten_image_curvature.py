
__all__ = ['weingarten_image_curvature']

import numpy as np

import ants
from ants.decorators import image_method
from ants.internal import get_lib_fn

@image_method
def weingarten_image_curvature(image, sigma=1.0, opt='mean'):
    """
    Uses the weingarten map to estimate image mean or gaussian curvature

    ANTsR function: `weingartenImageCurvature`
    
    Arguments
    ---------
    image : ANTsImage
        image from which curvature is calculated
    
    sigma : scalar
        smoothing parameter
    
    opt : string
        mean by default, otherwise `gaussian` or `characterize`
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('mni')).resample_image((3,3,3))
    >>> imagecurv = ants.weingarten_image_curvature(image)
    """
    if image.dimension not in {2,3}:
        raise ValueError('image must be 2D or 3D')

    if image.dimension == 2:
        d = image.shape
        temp = np.zeros(list(d)+[10])
        for k in range(1,7):
            voxvals = image[:d[0],:d[1]].numpy()
            temp[:d[0],:d[1],k] = voxvals
        temp = ants.from_numpy(temp)
        myspc = image.spacing
        myspc = list(myspc) + [min(myspc)]
        temp.set_spacing(myspc)
        temp = temp.clone('float')
    else:
        temp = image.clone('float')

    optnum = 0
    if opt == 'gaussian': 
        optnum = 6
    if opt == 'characterize':
        optnum = 5

    libfn = get_lib_fn('weingartenImageCurvature')
    mykout = libfn(temp.pointer, sigma, optnum)
    mykout = ants.from_pointer(mykout)
    if image.dimension == 3:
        return mykout
    elif image.dimension == 2:
        subarr = ants.from_numpy(mykout.numpy()[:,:,4])
        return ants.copy_image_info(image, subarr)
