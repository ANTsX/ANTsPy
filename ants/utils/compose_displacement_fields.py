
__all__ = ['compose_displacement_fields']

from ..core import ants_image as iio
from .. import utils


def compose_displacement_fields(displacement_field, 
                                warping_field):
    """
    Compose displacement fields.

    Arguments
    ---------
    displacement_field : ANTsImage displacement field
        displacement field

    warping_field : ANTsImage displacement field
        warping field


    Example
    -------
    >>> import ants
    """

    libfn = utils.get_lib_fn('composeDisplacementFieldsD%i' % displacement_field.dimension)
    comp_field = libfn(displacement_field.pointer, warping_field.pointer)

    new_image = iio.ANTsImage(pixeltype='float', dimension=displacement_field.dimension, 
                         components=displacement_field.dimension, pointer=comp_field).clone('float')
    return new_image


