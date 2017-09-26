
 

__all__ = ['create_jacobian_determinant_image']

from tempfile import mktemp

from ..core import ants_image as iio
from ..core import ants_image_io as iio2

from .. import utils


def create_jacobian_determinant_image(domain_image, tx, do_log=False, geom=False):
    """
    Compute the jacobian determinant from a transformation file
   
    ANTsR function: `createJacobianDeterminantImage`

    Arguments
    ---------
    domain_image : ANTsImage
        image that defines transformation domain
    
    tx : string
        deformation transformation file name
    
    do_log : boolean
        return the log jacobian
    
    geom : bolean
        use the geometric jacobian calculation (boolean)
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16'))
    >>> mi = ants.image_read( ants.get_ants_data('r64'))
    >>> fi = ants.resample_image(fi,(128,128),1,0)
    >>> mi = ants.resample_image(mi,(128,128),1,0)
    >>> mytx = ants.registration(fixed=fi , moving=mi, type_of_transform = ('SyN') )
    >>> jac = ants.create_jacobian_determinant_image(fi,mytx['fwdtransforms'][0],1)
    """
    dim = domain_image.dimension
    if isinstance(tx, iio.ANTsImage):
        txuse = mktemp(suffix='.nii.gz')
        iio2.image_write(tx, txuse)
    else:
        txuse = tx
    #args = [dim, txuse, do_log]
    dimage = domain_image.clone('double')
    args2 = [dim, txuse, dimage, int(do_log), int(geom)]
    processed_args = utils._int_antsProcessArguments(args2)
    libfn = utils.get_lib_fn('CreateJacobianDeterminantImage')
    libfn(processed_args)
    jimage = args2[2].clone('float')
    
    return jimage


