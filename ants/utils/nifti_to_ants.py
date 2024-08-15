__all__ = ["nifti_to_ants", "ants_to_nifti", "from_nibabel", "to_nibabel"]

from typing import TYPE_CHECKING

import numpy as np

import ants

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image


def nifti_to_ants(nib_image: "Nifti1Image"):
    """
    Convert a Nifti image to an ANTsPy image.

    Parameters
    ----------
    nib_image : Nifti1Image
        The Nifti image to be converted.

    Returns
    -------
    ants_image : ants.ANTsImage
        The converted ANTs image.
    """
    ndim = nib_image.ndim

    if ndim < 3:
        raise NotImplementedError(
            "Conversion is only implemented for 3D or higher images."
        )
    q_form = nib_image.get_qform()
    spacing = nib_image.header["pixdim"][1 : ndim + 1]

    origin = np.zeros(ndim)
    origin[:3] = np.dot(np.diag([-1, -1, 1]), q_form[:3, 3])

    direction = np.eye(ndim)
    direction[:3, :3] = np.dot(np.diag([-1, -1, 1]), q_form[:3, :3]) / spacing[:3]

    ants_img = ants.from_numpy(
        data=nib_image.get_fdata(),
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        direction=direction,
    )
    "add nibabel conversion (lacey import to prevent forced dependency)"

    return ants_img


def get_ras_affine_from_ants(ants_img) -> np.ndarray:
    """
    Convert ANTs image affine to RAS coordinate system.
    Source: https://github.com/fepegar/torchio/blob/main/src/torchio/data/io.py
    Parameters
    ----------
    ants_img : ants.ANTsImage
        The ANTs image whose affine is to be converted.

    Returns
    -------
    affine : np.ndarray
        The affine matrix in RAS coordinates.
    """
    spacing = np.array(ants_img.spacing)
    direction_lps = np.array(ants_img.direction)
    origin_lps = np.array(ants_img.origin)
    direction_length = direction_lps.shape[0] * direction_lps.shape[1]
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # 2D case (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # Fix potential bad NIfTI
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise NotImplementedError(f"Unexpected direction length = {direction_length}.")

    rotation_ras = np.dot(np.diag([-1, -1, 1]), rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(np.diag([-1, -1, 1]), origin_lps)

    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras

    return affine


def ants_to_nifti(img, header=None) -> "Nifti1Image":
    """
    Convert an ANTs image to a Nifti image.

    Parameters
    ----------
    img : ants.ANTsImage
        The ANTs image to be converted.
    header : Nifti1Header, optional
        Optional header to use for the Nifti image.

    Returns
    -------
    img : Nifti1Image
        The converted Nifti image.
    """
    from nibabel.nifti1 import Nifti1Image

    affine = get_ras_affine_from_ants(img)
    arr = img.numpy()

    if header is not None:
        header.set_data_dtype(arr.dtype)

    return Nifti1Image(arr, affine, header)


# Legacy names for backwards compatibility
from_nibabel = nifti_to_ants
to_nibabel = ants_to_nifti

if __name__ == "__main__":
    import nibabel as nib

    fn = ants.get_ants_data("mni")
    ants_img = ants.image_read(fn)
    nii_mni: "Nifti1Image" = nib.load(fn)
    ants_mni = to_nibabel(ants_img)
    assert (ants_mni.get_qform() == nii_mni.get_qform()).all()
    assert (ants_mni.affine == nii_mni.affine).all()
    temp = from_nibabel(nii_mni)

    assert ants.image_physical_space_consistency(ants_img, temp)

    fn = ants.get_data("ch2")
    ants_mni = ants.image_read(fn)
    nii_mni = nib.load(fn)
    ants_mni = to_nibabel(ants_mni)
    assert (ants_mni.get_qform() == nii_mni.get_qform()).all()

    nii_org = nib.load(fn)
    ants_org = ants.image_read(fn)
    temp = ants_org
    for i in range(10):
        temp = to_nibabel(ants_org)
        assert (temp.get_qform() == nii_org.get_qform()).all()
        assert (ants_mni.affine == nii_mni.affine).all()
        temp = from_nibabel(temp)
        assert ants.image_physical_space_consistency(ants_org, temp)
    for i in range(10):
        temp = from_nibabel(nii_org)
        assert ants.image_physical_space_consistency(ants_org, temp)
        temp = to_nibabel(temp)

        assert (temp.get_qform() == nii_org.get_qform()).all()
        assert (ants_mni.affine == nii_mni.affine).all()
