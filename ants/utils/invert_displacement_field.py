
__all__ = ['invert_displacement_field']

from ..core import ants_image as iio
from .. import utils


def invert_displacement_field(displacement_field, 
                              inverse_field_initial_estimate, 
                              maximum_number_of_iterations=20, 
                              mean_error_tolerance_threshold=0.001, 
                              max_error_tolerance_threshold=0.1,
                              enforce_boundary_condition=True):
    """
    Invert displacement field.

    Arguments
    ---------
    displacement_field : ANTsImage displacement field
        displacement field

    inverse_field_initial_estimate : ANTsImage displacement field
        initial guess

    maximum_number_of_iterations : integer
        number of iterations

    mean_error_tolerance_threshold : float
        mean error tolerance threshold

    max_error_tolerance_threshold : float
        max error tolerance threshold

    enforce_boundary_condition : bool
        enforce stationary boundary condition
        

    Example
    -------
    >>> import ants
    """

    libfn = utils.get_lib_fn('invertDisplacementFieldD%i' % displacement_field.dimension)
    inverse_field = libfn(displacement_field.pointer, inverse_field_initial_estimate.pointer, 
        maximum_number_of_iterations, mean_error_tolerance_threshold, 
        max_error_tolerance_threshold, enforce_boundary_condition)

    new_image = iio.ANTsImage(pixeltype='float', dimension=displacement_field.dimension, 
                         components=displacement_field.dimension, pointer=inverse_field).clone('float')
    return new_image


