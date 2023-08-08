
 

__all__ = ['create_jacobian_determinant_image','deformation_gradient']

from tempfile import mktemp

from ..core import ants_image as iio
from ..core import ants_image_io as iio2

from .. import utils


def deformation_gradient( warp_image ):
    """
    Compute the deformation gradient from an image containing a warp (deformation)
   
    ANTsR function: `NA`

    Arguments
    ---------
    warp_image : ANTsImage
        image that defines the deformation field (vector pixels)

    Returns
    -------
    ANTsImage with dimension*dimension components indexed in order U_xyz, V_xyz, W_xyz 
        where U is the x-component of deformation and xyz are spatial.

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16'))
    >>> mi = ants.image_read( ants.get_ants_data('r64'))
    >>> fi = ants.resample_image(fi,(128,128),1,0)
    >>> mi = ants.resample_image(mi,(128,128),1,0)
    >>> mytx = ants.registration(fixed=fi , moving=mi, type_of_transform = ('SyN') )
    >>> dg = ants.deformation_gradient( ants.image_read( mytx['fwdtransforms'][0] ) )
    """
    if isinstance(warp_image, iio.ANTsImage):
        txuse = mktemp(suffix='.nii.gz')
        iio2.image_write(warp_image, txuse)
    else:
        txuse = warp_image
        warp_image=ants.image_read(txuse)
    if not isinstance(warp_image, iio.ANTsImage):
        raise RuntimeError("antsimage is required")
    writtenimage = mktemp(suffix='.nrrd')
    dimage = warp_image.split_channels()[0].clone('double')
    dim = dimage.dimension
    args2 = [dim, txuse, writtenimage, int(0), int(0), int(1)]
    # print(args2)
    processed_args = utils._int_antsProcessArguments(args2)
    libfn = utils.get_lib_fn('CreateJacobianDeterminantImage')
    libfn(processed_args)
    jimage = iio2.image_read(writtenimage) 
    import os
    os.remove( writtenimage )
    return jimage
    import numpy as np
    if not isinstance(warp_image, iio.ANTsImage):
        raise RuntimeError("antsimage is required")
    dim = warp_image.dimension
    warpnp=warp_image.numpy()
    tshp=warp_image.shape
    tdir=warp_image.direction
    spc = warp_image.spacing
    it=np.ndindex(tshp)
    # print("first we need to rotate the warp by the direction cosines")
    for i in it:
        warpnp[i]=np.dot( tdir,warpnp[i])
    # print("second get deformation gradient")
    dg = []
    for k in range(dim):
        if dim == 2:
            temp=np.stack( np.gradient( warpnp[...,k], spc[0], spc[1], axis=range(dim) ), axis=dim)
        if dim == 3:
            temp=np.stack( np.gradient( warpnp[...,k], spc[0], spc[1], spc[2], axis=range(dim) ), axis=dim)
        dg.append(temp)
    return dg
    dg = np.stack(dg,axis=dim+1)
    dg = np.reshape( dg, warp_image.shape + (dim*dim,))
    dg = iio2.from_numpy( dg, has_components=True )
    iio.copy_image_info( warp_image, dg )
    return dg



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

