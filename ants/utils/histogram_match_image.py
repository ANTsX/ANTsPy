
__all__ = ['histogram_match_image']

import math

from ..core import ants_image as iio
from .. import utils


def histogram_match_image(source_image, reference_image, number_of_histogram_bins=255, number_of_match_points=64, use_threshold_at_mean_intensity=False):
    """
    Histogram match source image to reference image.

    Arguments
    ---------
    source_image : ANTsImage
        source image

    reference_image : ANTsImage
        reference image

    number_of_histogram_bins : integer
        number of bins for source and reference histograms

    number_of_match_points : integer
        number of points for histogram matching

    use_threshold_at_mean_intensity : boolean
        see ITK description.

    Example
    -------
    >>> import ants
    >>> src_img = ants.image_read(ants.get_data('r16'))
    >>> ref_img = ants.image_read(ants.get_data('r64'))
    >>> src_ref = ants.histogram_match_image(src_img, ref_img)
    """

    inpixeltype = source_image.pixeltype
    ndim = source_image.dimension
    if source_image.pixeltype != 'float':
        source_image = source_image.clone('float')
    if reference_image.pixeltype != 'float':
        reference_image = reference_image.clone('float')

    libfn = utils.get_lib_fn('histogramMatchImageF%i' % ndim)
    itkimage = libfn(source_image.pointer, reference_image.pointer, number_of_histogram_bins, number_of_match_points, use_threshold_at_mean_intensity)

    new_image = iio.ANTsImage(pixeltype='float', dimension=ndim, 
                         components=source_image.components, pointer=itkimage).clone(inpixeltype)
    return new_image


