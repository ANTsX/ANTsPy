
__all__ = ['integrate_velocity_field']

from ..core import ants_image as iio
from .. import utils


def integrate_velocity_field(velocity_field,
                             lower_integration_bound=0.0,
                             upper_integration_bound=1.0,
                             number_of_integration_steps=10):
    """
    Integrate velocity field.

    Arguments
    ---------
    velocity_field : ANTsImage velocity field
        time-varying displacement field

    lower_integration_bound: float
        Lower time bound for integration in [0, 1]

    upper_integration_bound: float
        Upper time bound for integration in [0, 1]

    number_of_integation_steps: integer
        Number of integration steps used in the Runge-Kutta solution

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_data( "r16" ) )
    >>> mi = ants.image_read( ants.get_data( "r27" ) )
    >>> reg = ants.registration(fi, mi, "TV[2]")
    >>> velocity_field = ants.image_read(reg['velocityfield'][0])
    >>> field = ants.integrate_velocity_field(velocity_field, 0.0, 1.0, 10) 
    >>> temp=ants.apply_ants_transform_to_image( 
	    ants.transform_from_displacement_field( field ), mi, fi )
    """

    libfn = utils.get_lib_fn('integrateVelocityFieldD%i' % (velocity_field.dimension-1))
    integrated_field = libfn(velocity_field.pointer, lower_integration_bound,
        upper_integration_bound, number_of_integration_steps)

    new_image = iio.ANTsImage(pixeltype='float', dimension=(velocity_field.dimension-1),
                         components=(velocity_field.dimension-1), pointer=integrated_field).clone('float')
    return new_image


