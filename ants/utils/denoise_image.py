
 

__all__ = ['denoise_image']

from .. import lib
from . import process_args as pargs
from .get_mask import get_mask

def denoise_image(img, mask=None, shrink_factor=1, p=1, r=3, noise_model='Rician'):
    """
    Arguments
    ---------
    noise_model : string
        'Rician' or 'Gaussian'

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> img = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> # add fairly large salt and pepper noise
    >>> imgnoise = img + np.random.randn(*img.shape).astype('float32')*5
    >>> imgdenoise = ants.denoise_image(imgnoise, ants.get_mask(img))
    """
    inpixeltype = img.pixeltype
    outimg = img.clone('float')

    if mask is None:
        mask = get_mask(img)

    mydim = img.dimension
    myargs = {
        'd': mydim,
        'i': img,
        'n': noise_model,
        'x': mask.clone('unsigned char'),
        's': int(shrink_factor),
        'p': p,
        'r': r,
        'o': outimg,
        'v': 0
    }
    processed_args = pargs._int_antsProcessArguments(myargs)
    lib.DenoiseImage(processed_args)
    return outimg.clone(inpixeltype)