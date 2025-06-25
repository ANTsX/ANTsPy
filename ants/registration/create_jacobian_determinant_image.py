


__all__ = ['create_jacobian_determinant_image',
           'deformation_gradient']

from tempfile import NamedTemporaryFile

import ants
import numpy as np
from ants.internal import get_lib_fn, process_arguments


def deformation_gradient( warp_image, to_rotation=False, to_inverse_rotation=False, py_based=False ):
    """
    Compute the deformation gradient from an image containing a warp (deformation).

    This function now includes a highly optimized pure Python/NumPy implementation.

    ANTsR function: `NA`

    Arguments
    ---------
    warp_image : ANTsImage (or filename if not py_based)
        image that defines the deformation field (vector pixels)

    to_rotation : boolean
        maps deformation gradient to a rotation matrix using polar decomposition.

    to_inverse_rotation : boolean
        map the deformation gradient to a rotation matrix, and return its inverse.
        This is useful for reorienting tensors and vectors after resampling.

    py_based: boolean
        If True, uses the optimized pure Python/NumPy implementation. If False,
        uses the classic C++ backend.

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
    >>> warp = ants.image_read( mytx['fwdtransforms'][0] )
    >>> # Use the fast, optimized Python implementation
    >>> dg_py = ants.deformation_gradient( warp, py_based=True )
    >>> dg_rot_py = ants.deformation_gradient( warp, to_rotation=True, py_based=True )
    """
    if not py_based:
        # --- Original C++ based implementation ---
        if ants.is_image(warp_image):
            txuse = NamedTemporaryFile(suffix='.nii.gz', delete=False).name
            ants.image_write(warp_image, txuse)
        else:
            txuse = warp_image
            warp_image=ants.image_read(txuse)
        if not ants.is_image(warp_image):
            raise RuntimeError("antsimage is required")
        writtenimage = NamedTemporaryFile(suffix='.nii.gz', delete=False).name
        dimage = warp_image.split_channels()[0].clone('double')
        dim = dimage.dimension
        tshp = dimage.shape
        args2 = [dim, txuse, writtenimage, int(0), int(0), int(1)]
        processed_args = process_arguments(args2)
        libfn = get_lib_fn('CreateJacobianDeterminantImage')
        libfn(processed_args)
        dg = ants.image_read(writtenimage)
        if to_rotation or to_inverse_rotation:
            newshape = tshp + (dim,dim)
            dg = np.reshape( dg.numpy(), newshape )
            U, s, Vh = np.linalg.svd(dg)
            Z = U @ Vh
            dets = np.linalg.det(Z)
            reflection_mask = dets < 0
            Vh[reflection_mask, -1, :] *= -1
            Z[reflection_mask] = U[reflection_mask] @ Vh[reflection_mask]
            dg = Z
            if to_inverse_rotation:
                dg = np.transpose(dg, axes=(*range(dg.ndim - 2), dg.ndim - 1, dg.ndim - 2))
            newshape = tshp + (dim*dim,)
            dg = np.reshape( dg, newshape )
            dg = ants.from_numpy( dg, has_components=True )
            dg = ants.copy_image_info( dimage, dg )
        import os
        os.remove( writtenimage )
        return dg

    # --- Optimized Python/NumPy Implementation ---
    if py_based:
        if not ants.is_image(warp_image):
            raise RuntimeError("antsimage is required")
        dim = warp_image.dimension
        tshp = warp_image.shape
        tdir = warp_image.direction
        spc = warp_image.spacing
        warpnp = warp_image.numpy()
        gradient_list = [np.gradient(warpnp[..., k], *spc, axis=range(dim)) for k in range(dim)]
        # This correctly calculates J.T, where dg[..., i, j] = d(u_j)/d(x_i)
        dg = np.stack([np.stack(grad_k, axis=-1) for grad_k in gradient_list], axis=-1)
        dg = (tdir @ dg).swapaxes(-1, -2)
        dg += np.eye(dim)
        if to_rotation or to_inverse_rotation:
            U, s, Vh = np.linalg.svd(dg)
            Z = U @ Vh
            dets = np.linalg.det(Z)
            reflection_mask = dets < 0
            Vh[reflection_mask, -1, :] *= -1
            Z[reflection_mask] = U[reflection_mask] @ Vh[reflection_mask]
            dg = Z
            if to_inverse_rotation:
                dg = np.transpose(dg, axes=(*range(dg.ndim - 2), dg.ndim - 1, dg.ndim - 2))
        new_shape = tshp + (dim * dim,)
        dg_reshaped = np.reshape(dg, new_shape)
        return ants.from_numpy(dg_reshaped, origin=warp_image.origin,
                            spacing=warp_image.spacing, direction=warp_image.direction,
                            has_components=True)




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

    geom : boolean
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
        txuse = NamedTemporaryFile(suffix='.nii.gz', delete=False).name
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

