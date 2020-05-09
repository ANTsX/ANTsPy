__all__ = ["simulate_displacement_field"]

import numpy as np

from ..core import ants_image as iio
from .. import utils


def simulate_displacement_field(domain_image, 
                                field_type="bspline", 
                                number_of_random_points=1000, 
                                sd_noise=10.0,
                                enforce_stationary_boundary=True,
                                number_of_fitting_levels=4,
                                mesh_size=1,
                                sd_smoothing=4.0):
    """
    simulate displacement field using either b-spline or exponential transform

    ANTsR function: `simulateDisplacementField`

    Arguments
    ---------
    domain_image : ANTsImage
        Domain image

    field_type : string
        Either "bspline" or "exponential".

    number_of_random_points : integer
        Number of displacement points.

    sd_noise : float
        Standard deviation of the displacement field noise.

    enforce_stationary_boundary : boolean
        Determines fixed boundary conditions.

    number_of_fitting_levels : integer
        Number of fitting levels (b-spline only).

    mesh_size : integer or n-D tuple
        Determines fitting resolution at base level (b-spline only).

    sd_smoothing : float
        Standard deviation of the Gaussian smoothing in mm (exponential only).
        
    Returns
    -------
    ANTs vector image.

    Example
    -------
    >>> import ants
    >>> domain = ants.image_read( ants.get_ants_data('r16'))
    >>> exp_field = ants.simulate_displacement_field(domain, field_type="exponential")
    >>> bsp_field = ants.simulate_displacement_field(domain, field_type="bspline")
    >>> bsp_xfrm = ants.transform_from_displacement_field(bsp_field * 3)
    >>> domain_warped = ants.apply_ants_transform_to_image(bsp_xfrm, domain, domain)
    """

    image_dimension = domain_image.dimension

    if field_type == 'bspline':
        if isinstance(mesh_size, int) == False and len(mesh_size) != image_dimension:
            raise ValueError("Incorrect specification for mesh_size.")

        spline_order = 3
        number_of_control_points = mesh_size + spline_order

        if isinstance(number_of_control_points, int) == True:
            number_of_control_points = np.repeat(number_of_control_points, image_dimension)

        libfn = utils.get_lib_fn("simulateBsplineDisplacementField%iD" % image_dimension)
        field = libfn(domain_image.pointer, number_of_random_points, sd_noise, 
                      enforce_stationary_boundary, number_of_fitting_levels, number_of_control_points)
        bspline_field = iio.ANTsImage(pixeltype='float', 
            dimension=image_dimension, components=image_dimension,
            pointer=field).clone('float')
        return bspline_field

    elif field_type == 'exponential':
        libfn = utils.get_lib_fn("simulateExponentialDisplacementField%iD" % image_dimension)
        field = libfn(domain_image.pointer, number_of_random_points, sd_noise, 
                      enforce_stationary_boundary, sd_smoothing)
        exp_field = iio.ANTsImage(pixeltype='float', 
            dimension=image_dimension, components=image_dimension,
            pointer=field).clone('float')
        return exp_field

    else:  
        raise ValueError("Unrecognized field type.")
