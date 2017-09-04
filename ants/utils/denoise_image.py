
 

__all__ = ['denoise_image']

from .. import lib
from . import process_args as pargs
from .get_mask import get_mask

def denoise_image(img, mask=None, shrink_factor=1, p=1, r=3, noise_model='Rician'):
    """
    Denoise an image using a spatially adaptive filter originally described in 
    J. V. Manjon, P. Coupe, Luis Marti-Bonmati, D. L. Collins, and M. Robles. 
    Adaptive Non-Local Means Denoising of MR Images With Spatially Varying 
    Noise Levels, Journal of Magnetic Resonance Imaging, 31:192-203, June 2010.

    ANTsR function: `denoiseImage`

    Arguments
    ---------
    img : ANTsImage
        scalar image to denoise.
    
    mask : ANTsImage
        to limit the denoise region.
    
    shrink_factor : scalar   
        downsampling level performed within the algorithm.
    
    p : integer
        patch radius for local sample.
    
    r : integer
        search radius from which to choose extra local samples.
    
    noise_model : string
        'Rician' or 'Gaussian'
    
    Returns
    -------
    ANTsImage

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