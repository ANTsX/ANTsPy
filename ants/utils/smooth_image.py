
 

__all__ = ['smooth_image']

import math

from .. import lib
from .. import utils
from ..core import ants_image as iio


_smooth_image_dict = {
    2: 'SmoothImage2D',
    3: 'SmoothImage3D',
    4: 'SmoothImage4D'
}


def _smooth_image_helper(img, sigma, sigma_in_physical_coordinates=True, FWHM=False, max_kernel_width=70):
    outimg = img.clone()
    if not isinstance(sigma, (tuple,list)):
        sigma = [sigma]

    if isinstance(sigma, (tuple, list)) and ((len(sigma) != img.dimension) and (len(sigma) != 1)):
        raise ValueError('Length of sigma must be either 1 or the dimensionality of input image')

    img_float = img.clone('float')
    if FWHM:
        sigma = [s/2.355 for s in sigma]

    max_kernel_width = int(math.ceil(max_kernel_width))

    smooth_image_fn = lib.__dict__[_smooth_image_dict[img.dimension]]
    outimg = smooth_image_fn(img_float._img, sigma, sigma_in_physical_coordinates, max_kernel_width)
    ants_outimg = iio.ANTsImage(outimg)
    return ants_outimg


def smooth_image(img, sigma, sigma_in_physical_coordinates=True, FWHM=False, max_kernel_width=32):
    """
    Smooth an image

    ANTsR function: `smoothImage`

    Arguments
    ---------
    img   
        Image to smooth
    
    sigma   
        Smoothing factor. Can be scalar, in which case the same sigma is applied to each dimension, or a vector of length dim(inimg) to specify a unique smoothness for each dimension.
    
    sigma_in_physical_coordinates : boolean  
        If true, the smoothing factor is in millimeters; if false, it is in pixels.
    
    FWHM : boolean    
        If true, sigma is interpreted as the full-width-half-max (FWHM) of the filter, not the sigma of a Gaussian kernel.
    
    max_kernel_width : scalar    
        Maximum kernel width
    
    Returns
    -------
    ANTsImage
    
    Example
    -------
    >>> img = ants.image_read( ants.get_ants_data('r16'))
    >>> simg = ants.smooth_image(img, (1.2,1.5))
    """
    if img.components == 1:
        return _smooth_image_helper(img, sigma, sigma_in_physical_coordinates, FWHM, max_kernel_width)
    else:
        imglist = utils.split_channels(img)
        newimgs = []
        for img in imglist:
            newimg = _smooth_image_helper(img, sigma, sigma_in_physical_coordinates, FWHM, max_kernel_width)
            newimgs.append(newimg)
        return utils.merge_channels(newimgs)

