
 

__all__ = ['denoise_image']

from .. import utils
from . import process_args as pargs
from .get_mask import get_mask

def denoise_image(image, mask=None, shrink_factor=1, p=1, r=3, noise_model='Rician'):
    """
    Denoise an image using a spatially adaptive filter originally described in 
    J. V. Manjon, P. Coupe, Luis Marti-Bonmati, D. L. Collins, and M. Robles. 
    Adaptive Non-Local Means Denoising of MR Images With Spatially Varying 
    Noise Levels, Journal of Magnetic Resonance Imaging, 31:192-203, June 2010.

    ANTsR function: `denoiseImage`

    Arguments
    ---------
    image : ANTsImage
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
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> # add fairly large salt and pepper noise
    >>> imagenoise = image + np.random.randn(*image.shape).astype('float32')*5
    >>> imagedenoise = ants.denoise_image(imagenoise, ants.get_mask(image))
    """
    inpixeltype = image.pixeltype
    outimage = image.clone('float')

    if mask is None:
        mask = get_mask(image)

    mydim = image.dimension
    myargs = {
        'd': mydim,
        'i': image,
        'n': noise_model,
        'x': mask.clone('unsigned char'),
        's': int(shrink_factor),
        'p': p,
        'r': r,
        'o': outimage,
        'v': 0
    }
    processed_args = pargs._int_antsProcessArguments(myargs)
    libfn = utils.get_lib_fn('DenoiseImage')
    libfn(processed_args)
    return outimage.clone(inpixeltype)


