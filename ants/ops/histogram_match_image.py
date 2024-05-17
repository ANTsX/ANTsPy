
__all__ = ['histogram_match_image',
           'histogram_match_image2']

import numpy as np

import ants
from ants.decorators import image_method
from ants.internal import get_lib_fn


@image_method
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

    libfn = get_lib_fn('histogramMatchImageF%i' % ndim)
    itkimage = libfn(source_image.pointer, reference_image.pointer, number_of_histogram_bins, number_of_match_points, use_threshold_at_mean_intensity)

    new_image = ants.from_pointer(itkimage).clone(inpixeltype)
    return new_image

@image_method
def histogram_match_image2(source_image, reference_image, 
                           source_mask=None, reference_mask=None,
                           match_points=64,
                           transform_domain_size=255):
    """
    Transform image intensities based on histogram mapping.

    Apply B-spline 1-D maps to an input image for intensity warping.

    Arguments
    ---------
    source_image : ANTsImage
        source image

    reference_image : ANTsImage
        reference image

    source_mask : ANTsImage
        source mask

    reference_mask : ANTsImage
        reference mask
                
    match_points : integer or tuple
        Parametric points at which the intensity transform displacements are 
        specified between [0, 1], i.e. quantiles.  Alternatively, a single number 
        can be given and the sequence is linearly spaced in [0, 1]. 

    transform_domain_size : integer
        Defines the sampling resolution of the B-spline warping.

    Returns
    -------
    ANTs image

    Example
    -------
    >>> import ants
    >>> src_img = ants.image_read(ants.get_data('r16'))
    >>> ref_img = ants.image_read(ants.get_data('r64'))
    >>> src_ref = ants.histogram_match_image2(src_img, ref_img)
    """

    if not isinstance(match_points, int):
        if any(b < 0 for b in match_points) and any(b > 1 for b in match_points):
            raise ValueError("If specifying match_points as a vector, values must be in the range [0, 1]")

    # Use entire image if mask isn't specified
    if source_mask is None:
        source_mask = source_image * 0 + 1
    if reference_mask is None:
        reference_mask = reference_image * 0 + 1

    source_array = source_image.numpy()
    source_mask_array = source_mask.numpy()
    source_masked_min = source_image[source_mask != 0].min()
    source_masked_max = source_image[source_mask != 0].max()

    reference_array = reference_image.numpy()
    reference_mask_array = reference_mask.numpy()

    parametric_points = None
    if not isinstance(match_points, int):
        parametric_points = match_points
    else:
        parametric_points = np.linspace(0, 1, match_points)

    source_intensity_quantiles = np.quantile(source_array[source_mask_array != 0], parametric_points)
    reference_intensity_quantiles = np.quantile(reference_array[reference_mask_array != 0], parametric_points)
    displacements = reference_intensity_quantiles - source_intensity_quantiles

    scattered_data = np.reshape(displacements, (len(displacements), 1))
    parametric_data = np.reshape(parametric_points * (source_masked_max - source_masked_min) + source_masked_min, (len(parametric_points), 1))

    transform_domain_origin = source_masked_min
    transform_domain_spacing = (source_masked_max - transform_domain_origin) / (transform_domain_size - 1)

    bspline_histogram_transform = ants.fit_bspline_object_to_scattered_data(scattered_data,
        parametric_data, [transform_domain_origin], [transform_domain_spacing], [transform_domain_size],
        data_weights=None, is_parametric_dimension_closed=None, number_of_fitting_levels=8, 
        mesh_size=1, spline_order=3)

    transform_domain = np.linspace(source_masked_min, source_masked_max, transform_domain_size)

    transformed_source_array = source_image.numpy()
    for i in range(len(transform_domain) - 1):
        indices = np.where((source_array >= transform_domain[i]) & (source_array < transform_domain[i+1]))
        intensities = source_array[indices]

        alpha = (intensities - transform_domain[i])/(transform_domain[i+1] - transform_domain[i])
        xfrm = alpha * (bspline_histogram_transform[i+1] - bspline_histogram_transform[i]) + bspline_histogram_transform[i]
        transformed_source_array[indices] = intensities + xfrm

    transformed_source_image = ants.from_numpy(transformed_source_array, origin=source_image.origin,
        spacing=source_image.spacing, direction=source_image.direction)
    transformed_source_image[source_mask == 0] = source_image[source_mask == 0]

    return(transformed_source_image)

