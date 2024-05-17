
 

__all__ = ['create_jacobian_determinant_image',
           'deformation_gradient']

from tempfile import mktemp

import ants
from ants.internal import get_lib_fn, process_arguments


def deformation_gradient( warp_image, to_rotation=False, py_based=False ):
    """
    Compute the deformation gradient from an image containing a warp (deformation)
   
    ANTsR function: `NA`

    Arguments
    ---------
    warp_image : ANTsImage (or filename if not py_based)
        image that defines the deformation field (vector pixels)

    to_rotation : boolean maps deformation gradient to a rotation matrix

    py_based: boolean uses pure python implementation (maybe slow)

    Returns
    -------
    ANTsImage with dimension*dimension components indexed in order U_xyz, V_xyz, W_xyz 
        where U is the x-component of deformation and xyz are spatial.

    Note
    -------
    the to_rotation option is still experimental. use with caution.

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
    import numpy as np
    def polar_decomposition(X):
         U, d, V = np.linalg.svd(X, full_matrices=False)
         P = np.matmul(U, np.matmul(np.diag(d), np.transpose(U)))
         Z = np.matmul(U, V)
         if np.linalg.det(Z) < 0:
             n = X.shape[0]
             reflection_matrix = np.identity(n)
             reflection_matrix[0,0] = -1.0
             Z = np.matmul(Z, reflection_matrix)
         return({"P" : P, "Z" : Z, "Xtilde" : np.matmul(P, Z)})
    if not py_based:
        if ants.is_image(warp_image):
            txuse = mktemp(suffix='.nii.gz')
            ants.image_write(warp_image, txuse)
        else:
            txuse = warp_image
            warp_image=ants.image_read(txuse)
        if not ants.is_image(warp_image):
            raise RuntimeError("antsimage is required")
        writtenimage = mktemp(suffix='.nrrd')
        dimage = warp_image.split_channels()[0].clone('double')
        dim = dimage.dimension
        tshp = dimage.shape
        args2 = [dim, txuse, writtenimage, int(0), int(0), int(1)]
        processed_args = process_arguments(args2)
        libfn = get_lib_fn('CreateJacobianDeterminantImage')
        libfn(processed_args)
        dg = ants.image_read(writtenimage) 
        if to_rotation:
            newshape = tshp + (dim,dim)
            dg = np.reshape( dg.numpy(), newshape )
            it=np.ndindex(tshp)
            for i in it:
                dg[i]=polar_decomposition( dg[i] )['Z']
            newshape = tshp + (dim*dim,)
            dg = np.reshape( dg, newshape )
            dg = ants.from_numpy( dg, has_components=True )
            dg = ants.copy_image_info( dimage, dg )
        import os
        os.remove( writtenimage )
        return dg
    if py_based:
        if not ants.is_image(warp_image):
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
        dg = np.stack(dg,axis=dim+1)
        it=np.ndindex(tshp)
        ident = np.eye( dim )
        for i in it:
            dg[i]=dg[i]+ident
        if to_rotation:
            it=np.ndindex(tshp)
            for i in it:
                dg[i]=polar_decomposition( dg[i] )['Z']
        newshape = tshp + (dim*dim,)
        dg = np.reshape( dg, newshape )
        dg = ants.from_numpy( dg, has_components=True )
        dg = ants.copy_image_info( warp_image, dg )
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
    if ants.is_image(tx):
        txuse = mktemp(suffix='.nii.gz')
        ants.image_write(tx, txuse)
    else:
        txuse = tx
    #args = [dim, txuse, do_log]
    dimage = domain_image.clone('double')
    args2 = [dim, txuse, dimage, int(do_log), int(geom)]
    processed_args = process_arguments(args2)
    libfn = get_lib_fn('CreateJacobianDeterminantImage')
    libfn(processed_args)
    jimage = args2[2].clone('float')
    
    return jimage

