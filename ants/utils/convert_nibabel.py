
__all__ = ['to_nibabel', 'from_nibabel']

import os
from tempfile import mktemp
import numpy as np
from ..core import ants_image_io as iio2


def to_nibabel(image):
    """
    Convert an ANTsImage to a Nibabel image
    """
    if image.dimension != 3:
        raise ValueError('Only 3D images currently supported')

    import nibabel as nib
    array_data = image.numpy()
    affine = np.hstack([image.direction*np.diag(image.spacing),np.array(image.origin).reshape(3,1)])
    affine = np.vstack([affine, np.array([0,0,0,1.])])
    affine[:2,:] *= -1
    new_img = nib.Nifti1Image(array_data, affine)
    return new_img


def from_nibabel(nib_image):
    """
    Convert a nibabel image to an ANTsImage
    """
    tmpfile = mktemp(suffix='.nii.gz')
    nib_image.to_filename(tmpfile)
    new_img = iio2.image_read(tmpfile)
    os.remove(tmpfile)
    return new_img
