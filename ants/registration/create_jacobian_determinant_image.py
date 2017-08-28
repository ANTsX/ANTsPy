
 

__all__ = ['create_jacobian_determinant_image']

from tempfile import mktemp

from ..core import ants_image as iio
from ..core import image_io as iio2

from .. import utils
from .. import lib


def create_jacobian_determinant_image(domain_img, tx, do_log=False, geom=False):
    """
    Create Jacobian Determinant image
    
    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16') ).clone('float')
    >>> fi = ants.n3_bias_field_correction(fi, 2)
    >>> mi = ants.image_read( ants.get_ants_data('r64') ).clone('float')
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
    >>> jac = ants.create_jacobian_determinant_image(fi, mytx['fwdtransforms'][0], 1)
    """
    dim = domain_img.dimension
    if isinstance(tx, iio.ANTsImage):
        txuse = mktemp(suffix='.nii.gz')
        iio2.image_write(tx, txuse)
    else:
        txuse = tx
    #args = [dim, txuse, do_log]
    dimg = domain_img.clone('double')
    args2 = [dim, txuse, dimg, int(do_log), int(geom)]
    processed_args = utils._int_antsProcessArguments(args2)
    lib.CreateJacobianDeterminantImage(processed_args)
    jimg = args2[2].clone('float')
    
    return jimg


