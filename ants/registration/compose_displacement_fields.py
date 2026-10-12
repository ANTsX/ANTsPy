
__all__ = ['compose_displacement_fields']

import ants
from ants.internal import get_lib_fn


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

    libfn = get_lib_fn('composeDisplacementFieldsD%i' % displacement_field.dimension)

    if displacement_field.pixeltype != 'float':
        displacement_field = displacement_field.clone('float')
    if warping_field.pixeltype != 'float':
        warping_field = warping_field.clone('float')

    comp_field = libfn(displacement_field.pointer, warping_field.pointer)

    new_image = ants.from_pointer(comp_field).clone('float')
    return new_image


