__all__ = ["histogram_warp_image_intensities"]

import ants
import numpy as np

def histogram_warp_image_intensities(image,
                                     break_points=(0.25, 0.5, 0.75),
                                     displacements=None,
                                     clamp_end_points=(False, False),
                                     sd_displacements=0.05,
                                     transform_domain_size=20):
    """
    Transform image intensities based on histogram mapping.

    Apply B-spline 1-D maps to an input image for intensity warping.

    Arguments
    ---------
    image : ANTsImage
        Input image.

    break_points : integer or tuple
        Parametric points at which the intensity transform displacements
        are specified between [0, 1].  Alternatively, a single number can
        be given and the sequence is linearly spaced in [0, 1].

    displacements : tuple
        displacements to define intensity warping.  Length must be equal to the
        breakPoints.  Alternatively, if None random displacements are chosen
        (random normal:  mean = 0, sd = sd_displacements).

    sd_displacements : float
        Characterize the randomness of the intensity displacement.

    clamp_end_points : 2-element tuple of booleans
        Specify non-zero intensity change at the ends of the histogram.

    transform_domain_size : integer
        Defines the sampling resolution of the B-spline warping.

    Returns
    -------
    ANTs image

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data("r64"))
    >>> transformed_image = histogram_warp_image_intensities( image )
    """

    if not len(clamp_end_points) == 2:
        raise ValueError("clamp_end_points must be a boolean tuple of length 2.")

    if not isinstance(break_points, int):
        if any(b < 0 for b in break_points) or any(b > 1 for b in break_points):
            raise ValueError("If specifying break_points as a vector, values must be in the range [0, 1]")

    parametric_points = None
    number_of_nonzero_displacements = 1
    if not isinstance(break_points, int):
        parametric_points = break_points
        number_of_nonzero_displacements = len(break_points)
        if clamp_end_points[0] is True:
            parametric_points = (0, *parametric_points)
        if clamp_end_points[1] is True:
            parametric_points = (*parametric_points, 1)
    else:
        total_number_of_break_points = break_points
        if clamp_end_points[0] is True:
            total_number_of_break_points += 1
        if clamp_end_points[1] is True:
            total_number_of_break_points += 1
        parametric_points = np.linspace(0, 1, total_number_of_break_points)
        number_of_nonzero_displacements = break_points

    if displacements is None:
        displacements = np.random.normal(loc=0.0, scale=sd_displacements, size=number_of_nonzero_displacements)

    weights = np.ones(len(displacements))
    if clamp_end_points[0] is True:
        displacements = (0, *displacements)
        weights = np.concatenate((1000 * np.ones(1), weights))
    if clamp_end_points[1] is True:
        displacements = (*displacements, 0)
        weights = np.concatenate((weights, 1000 * np.ones(1)))

    if not len(displacements) == len(parametric_points):
        raise ValueError("Length of displacements does not match the length of the break points.")

    scattered_data = np.reshape(displacements, (len(displacements), 1))
    parametric_data = np.reshape(parametric_points, (len(parametric_points), 1))

    transform_domain_origin = 0
    transform_domain_spacing = (1.0 - transform_domain_origin) / (transform_domain_size - 1)

    bspline_histogram_transform = ants.fit_bspline_object_to_scattered_data(scattered_data,
        parametric_data, [transform_domain_origin], [transform_domain_spacing], [transform_domain_size],
        data_weights=weights, is_parametric_dimension_closed=None, number_of_fitting_levels=4, 
        mesh_size=1, spline_order=3)

    transform_domain = np.linspace(0, 1, transform_domain_size)

    normalized_image = ants.iMath(ants.image_clone(image), "Normalize")
    transformed_array = normalized_image.numpy()
    normalized_array = normalized_image.numpy()

    for i in range(len(transform_domain) - 1):
        indices = np.where((normalized_array >= transform_domain[i]) & (normalized_array < transform_domain[i+1]))
        intensities = normalized_array[indices]

        alpha = (intensities - transform_domain[i])/(transform_domain[i+1] - transform_domain[i])
        xfrm = alpha * (bspline_histogram_transform[i+1] - bspline_histogram_transform[i]) + bspline_histogram_transform[i]
        transformed_array[indices] = intensities + xfrm

    transformed_image = (ants.from_numpy(transformed_array, origin=image.origin,
        spacing=image.spacing, direction=image.direction) * (image.max() - image.min())) + image.min()

    return(transformed_image)

