
 

__all__ = ['smooth_image']

import math

from ants.decorators import image_method
from ants.internal import get_lib_fn

import ants


@image_method
def smooth_image(image, sigma, sigma_in_physical_coordinates=True, FWHM=False, max_kernel_width=32):
    """
    Smooth an image

    ANTsR function: `smoothImage`

    Arguments
    ---------
    image   
        Image to smooth
    
    sigma   
        Smoothing factor. Can be scalar, in which case the same sigma is applied to each dimension, or a vector of length dim(inimage) to specify a unique smoothness for each dimension.
    
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
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16'))
    >>> simage = ants.smooth_image(image, (1.2,1.5))
    """
    if image.components == 1:
        outimage = image.clone()
        if not isinstance(sigma, (tuple,list)):
            sigma = [sigma]

        if isinstance(sigma, (tuple, list)) and ((len(sigma) != image.dimension) and (len(sigma) != 1)):
            raise ValueError('Length of sigma must be either 1 or the dimensionality of input image')

        image_float = image.clone('float')
        if FWHM:
            sigma = [s/2.355 for s in sigma]

        max_kernel_width = int(math.ceil(max_kernel_width))

        smooth_image_fn = get_lib_fn('SmoothImage')
        outimage = smooth_image_fn(image_float.pointer, sigma, sigma_in_physical_coordinates, max_kernel_width)
        ants_outimage = ants.from_pointer(outimage)
        return ants_outimage
    else:
        imagelist = ants.split_channels(image)
        newimages = []
        for image in imagelist:
            newimage = smooth_image(image, sigma, sigma_in_physical_coordinates, FWHM, max_kernel_width)
            newimages.append(newimage)
        return ants.merge_channels(newimages)

