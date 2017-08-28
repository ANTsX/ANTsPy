
 

__all__ = ['smooth_image']

import math

from .. import lib
from .. import utils
from ..core import ants_image as iio


_smooth_image_dict = {
    2: lib.SmoothImage2D,
    3: lib.SmoothImage3D,
    4: lib.SmoothImage4D
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

    smooth_image_fn = _smooth_image_dict[img.dimension]
    outimg = smooth_image_fn(img_float._img, sigma, sigma_in_physical_coordinates, max_kernel_width)
    ants_outimg = iio.ANTsImage(outimg)
    return ants_outimg


def smooth_image(img, sigma, sigma_in_physical_coordinates=True, FWHM=False, 
                max_kernel_width=32, median_filter=False, max_error=0.01):
    """
    Dev Note
    --------
    This ANTsPy function `smooth_image` may be slightly different than the ANTsR function `smoothImage`
    because this code calls the ANTs implementation while ANTsR has its own re-implementation.
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

