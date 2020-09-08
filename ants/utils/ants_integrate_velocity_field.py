__all__ = ["integrate_velocity_field"]

from .process_args import _int_antsProcessArguments
from .. import utils


def integrate_velocity_field(
    reference_image,
    velocity_field_filename,
    deformation_field_filename,
    time_0=0,
    time_1=1,
    delta_time=0.01,
):
    """
    Integrate a velocityfield

    ANTsR function: `integrateVelocityField`

    Arguments
    ---------
    reference_image : ANTsImage
        Reference image domain, same as velocity field space

    velocity_field_filename : scalar (optional)
        Lower edge of threshold window

    deformation_field_filename : scalar (optional)
        Higher edge of threshold window

    time_0 : scalar
        Typically one or zero but can take intermediate values

    time_1 : scalar
        Typically one or zero but can take intermediate values

    delta_time : scalar
        Time step value in zero to one; typically 0.01

    Returns
    -------
    None

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_data( "r16" ) )
    >>> mi = ants.image_read( ants.get_data( "r27" ) )
    >>> mytx2 = ants.registration( fi, mi, "TV[2]" )
    >>> ants.integrate_velocity_field( fi, mytx2$velocityfield,  "/tmp/def.nii.gz" )
    """
    args = [
        reference_image,
        velocity_field_filename,
        deformation_field_filename,
        time_0,
        time_1,
        delta_time,
    ]
    processed_args = _int_antsProcessArguments(args)
    libfn = utils.get_lib_fn("ANTSIntegrateVelocityField")
    libfn(processed_args)
