import numpy as np
import ants
from ants.decorators import image_method

__all__ = ['image_physical_space_consistency',
           'allclose']

@image_method
def image_physical_space_consistency(image1, image2, tolerance=1e-2, datatype=False):
    """
    Check if two or more ANTsImage objects occupy the same physical space

    ANTsR function: `antsImagePhysicalSpaceConsistency`

    Arguments
    ---------
    *images : ANTsImages
        images to compare

    tolerance : float
        tolerance when checking origin and spacing

    data_type : boolean
        If true, also check that the image data types are the same

    Returns
    -------
    boolean
        true if images share same physical space, false otherwise
    """
    images = [image1, image2]

    img1 = images[0]
    for img2 in images[1:]:
        if (not ants.is_image(img1)) or (not ants.is_image(img2)):
            raise ValueError('Both images must be of class `AntsImage`')

        # image dimension check
        if img1.dimension != img2.dimension:
            return False

        # image spacing check
        space_diffs = sum([abs(s1-s2)>tolerance for s1, s2 in zip(img1.spacing, img2.spacing)])
        if space_diffs > 0:
            return False

        # image origin check
        origin_diffs = sum([abs(s1-s2)>tolerance for s1, s2 in zip(img1.origin, img2.origin)])
        if origin_diffs > 0:
            return False

        # image direction check
        origin_diff = np.allclose(img1.direction, img2.direction, atol=tolerance)
        if not origin_diff:
            return False

        # data type
        if datatype == True:
            if img1.pixeltype != img2.pixeltype:
                return False

            if img1.components != img2.components:
                return False

    return True


@image_method
def allclose(image1, image2):
    """
    Check if two images have the same array values
    """
    return np.allclose(image1.numpy(), image2.numpy())