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

    velocity_field_filename : string
        Filename to velocity field, output from ants.registration

    deformation_field_filename : string
        Filename to output deformation field

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
    >>> ants.integrate_velocity_field( fi, mytx2['velocityfield'][0],  "/tmp/def.nii.gz", 0, 1, 0.01 )
    >>> mydef = ants.apply_transforms( fi, mi, ["/tmp/def.nii.gz", mytx2['fwdtransforms'][1]] )
    >>> ants.image_mutual_information(fi,mi)
    >>> ants.image_mutual_information(fi,mytx2['warpedmovout'])
    >>> ants.image_mutual_information(fi,mydef)
    >>> ants.integrate_velocity_field( fi, mytx2['velocityfield'][0],  "/tmp/defi.nii.gz", 1, 0, 0.5 )
    >>> mydefi = ants.apply_transforms( mi, fi, [ mytx2['fwdtransforms'][1], "/tmp/defi.nii.gz" ] )
    >>> ants.image_mutual_information(mi,mydefi)
    >>> ants.image_mutual_information(mi,mytx2['warpedfixout'])
    """

    libfn = utils.get_lib_fn("integrateVelocityField")
    if reference_image.dimension == 2:
        libfn.integrateVelocityField2D(
            reference_image.pointer,
            velocity_field_filename,
            deformation_field_filename,
            time_0,
            time_1,
            delta_time,
        )
    if reference_image.dimension == 3:
        libfn.integrateVelocityField3D(
            reference_image.pointer,
            velocity_field_filename,
            deformation_field_filename,
            time_0,
            time_1,
            delta_time,
        )
