__all__ = ["to_nibabel", "from_nibabel", "to_ants"]

import os
from tempfile import mkstemp
import numpy as np
import nibabel as nib
from ..core import ants_image_io as iio2


def to_nibabel(image):
    """
    Convert an ANTsImage to a Nibabel image
    """
    import nibabel as nib

    fd, tmpfile = mkstemp(suffix=".nii.gz")
    image.to_filename(tmpfile)
    new_img = nib.load(tmpfile)
    os.close(fd)
    # os.remove(tmpfile) ## Don't remove tmpfile as nibabel lazy loads the data.
    return new_img


def from_nibabel(nib_image):
    """
    Convert a nibabel image to an ANTsImage
    """
    fd, tmpfile = mkstemp(suffix=".nii.gz")
    nib_image.to_filename(tmpfile)
    new_img = iio2.image_read(tmpfile)
    os.close(fd)
    os.remove(tmpfile)
    return new_img


def to_ants(img: nib.Nifti1Image):
    """
    Converts a given Nifti image into an ANTsPy image

    Parameters
    ----------
        img: NiftiImage

    Returns
    -------
        ants_image: ANTsImage
    """
    ndim = img.ndim
    q_form = img.get_qform()
    spacing = img.header["pixdim"][1 : ndim + 1]

    origin = np.zeros((ndim))
    origin[:3] = q_form[:3, 3]

    direction = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    ants_img = ants.from_numpy(
        data=img.get_data().astype(np.float),
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        direction=direction,
    )

    return ants_img
