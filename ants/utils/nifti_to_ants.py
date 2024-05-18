__all__ = ["nifti_to_ants"]

import os
from tempfile import mkstemp
import numpy as np
import ants

def nifti_to_ants( nib_image ):
    """
    Converts a given Nifti image into an ANTsPy image

    Parameters
    ----------
        img: NiftiImage

    Returns
    -------
        ants_image: ANTsImage
    """
    ndim = nib_image.ndim

    if ndim < 3:
        print("Dimensionality is less than 3.")
        return None

    q_form = nib_image.get_qform()
    spacing = nib_image.header["pixdim"][1 : ndim + 1]

    origin = np.zeros((ndim))
    origin[:3] = q_form[:3, 3]

    direction = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    ants_img = ants.from_numpy(
        data = nib_image.get_data().astype( np.float ),
        origin = origin.tolist(),
        spacing = spacing.tolist(),
        direction = direction )
    
    return ants_img
