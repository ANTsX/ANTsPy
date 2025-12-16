__all__ = ["nibabel_nifti_to_ants"]

import numpy as np
import ants
from ants.core import ants_image

from .nifti_utils import deshear_affine_transform_matrix,_spacing_and_dirs_from_affine,_ras_to_lps_affine

def from_nibabel_nifti( nib_image, deshear_threshold=1e-6, max_angle_deviation=0.5, pixeltype='float'):
    """
    Converts a given nibabel Nifti image into an ANTsPy image. The nifti image's affine transform is used to set the origin,
    spacing, and direction of the ANTsPy image.


    Parameters
    ----------
    img: NiftiImage from nibabel. This must be a 3D, 4D (time series) or 5D (multi-component) image. The pixel datatype must be
    compatible with ANTsPy, one of "uint8", "uint32", "float32", "float64". Units should be mm for spatial dimensions and
    seconds for time.

    deshear_threshold: float
        Threshold for action to remove the shear component of affine transform. If shear is above this threshold, it will be
        removed. Otherwise, the affine will be used as-is.

    max_angle_deviation: float
        Maximum angle deviation in degrees for directions after deshearing the affine. If the desheared directions deviate
        from the originals by more than this value (in degrees), a ValueError is raised.

    pixeltype: str
        Pixel type for the output ANTsPy image. One of "uint8", "uint32", "float32", "float64".

    Returns
    -------
        ants_image: ANTsImage
    """
    ndim = nib_image.ndim

    # nibabel RAS+ affine transform
    A = nib_image.affine

    if ndim == 3:
        if nib_image.header.get_xyzt_units()[0] != 'mm':
            raise ValueError("3D image spatial units must be in mm")
    elif ndim == 4:
        if nib_image.header.get_xyzt_units() != ('mm', 'sec'):
            raise ValueError("4D image spatial units must be in mm and time units in seconds")
    elif ndim == 5:
        if nib_image.header.get_xyzt_units() != ('mm', 'sec'):
            raise ValueError("5D image spatial units must be in mm and time units in seconds")
    else:
        raise ValueError("Conversion only supported for 3D, 4D, or 5D images")

    Q = deshear_affine_transform_matrix(A, deshear_threshold, max_angle_deviation)

    # Convert to ITK LPS+ coordinates
    Q = _ras_to_lps_affine(Q)

    # Get spacing, origin from nifti image
    spacing, direction = _spacing_and_dirs_from_affine(Q[:3, :3])

    origin = Q[:3, 3].tolist()

    zooms = nib_image.header.get_zooms()

    for i in range(3):
        if not np.isclose(spacing[i], zooms[i]):
            raise ValueError(
                f"Extracted spacing {spacing[i]} does not match header zooms {zooms[i]} along dimension {i}"
            )

    itk_to_np_type_map = {
        "unsigned char": np.uint8,
        "unsigned int": np.uint32,
        "float": np.float32,
        "double": np.float64,
    }

    if pixeltype not in itk_to_np_type_map:
        raise ValueError(f"Unsupported pixeltype {pixeltype}. Supported types are: {list(itk_to_np_type_map.keys())}")

    fdata = nib_image.get_fdata(dtype=itk_to_np_type_map[pixeltype])

    # If we have a 4D image we need extra information
    if ndim == 4:
        # 4D spacing is just the 3D spacing plus time spacing
        time_spacing = zooms[3]
        spacing.append(time_spacing)

        # 4D direction is just the 3D direction plus identity for time dimension
        direction_4d = np.eye(4)
        direction_4d[:3, :3] = direction
        direction = direction_4d

        # 4D origin is just the 3D origin plus any time origin
        origin.append(float(nib_image.header['toffset']))

    has_components = False

    if ndim == 5:
        # if 5D, fdata will be a 4D image with dim[4] = 1 and dim[5] = number of components
        # reduce this to a 3D image with components in the 4th dimension for ANTsPy
        has_components = True
        fdata = np.squeeze(fdata, axis=3)

    ants_img = ants.from_numpy(
        data = fdata,
        origin = origin,
        spacing = spacing,
        direction = direction,
        has_components = has_components)

    return ants_img


def to_nibabel_nifti( ants_image, header=None):
    """
    Converts a given ANTsPy image into a nibabel Nifti image.

    Parameters
    ----------
    ants_image: ANTsImage
        The input ANTsPy image to be converted. Should be a 3D, 4D (time series) or 5D (multi-component) image.

    header: nibabel Nifti1Header, optional
        An optional Nifti1Header to be used for the output Nifti image. If the transform of this header differs from the input
        image, the header qform will be removed, the input image transform will be used as the sform, and the sform code will
        be set to 'aligned'.

    Returns
    -------
        nib_image: NiftiImage from nibabel
    """
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Image

    if ants_image.dimension < 3 or ants_image.dimension > 5:
        raise ValueError("Conversion only supported for 3D, 4D, or 5D images")

    data = ants_image.numpy()

    if ants_image.has_components:
        # nibabel expects components in the 5th dimension, so array has to have dims (x,y,z,1,components)
        data = np.expand_dims(data, axis=3)

    spacing = ants_image.spacing
    direction = ants_image.direction
    origin = ants_image.origin

    # Construct affine matrix
    affine = np.eye(4)
    for i in range(3):
        for j in range(3):
            affine[i, j] = direction[i][j] * spacing[j]
        affine[i, 3] = origin[i]

    # Convert from ITK LPS+ to RAS+ (same operation as RAS+ to LPS+, different input)
    affine = _ras_to_lps_affine(affine)

    if header is not None:
        header.set_data_dtype(data.dtype)

    nib_image = Nifti1Image(data, affine, header=header)

    nib_image.header.set_xyzt_units('mm', 'sec')

    if ants_image.dimension == 4:
        nib_image.header['toffset'] = origin[3]
        zooms = list(nib_image.header.get_zooms())
        zooms[3] = spacing[3]
        nib_image.header.set_zooms(zooms)

    return nib_image
