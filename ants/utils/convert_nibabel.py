__all__ = ["to_nibabel", "from_nibabel"]

import os
from tempfile import mkstemp
import numpy as np
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
