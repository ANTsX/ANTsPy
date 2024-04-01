__all__ = ["label_overlap_measures"]

import pandas as pd

from .. import utils


def label_overlap_measures(source_image, target_image):
    """
    Get overlap measures from two label images (e.g., Dice)

    ANTsR function: `labelOverlapMeasures`

    Arguments
    ---------
    source image : ANTsImage
        Source image

    target_image : ANTsImage
        Target image

    Returns
    -------
    data frame with measures for each label and all labels combined

    Example
    -------
    >>> import ants
    >>> r16 = ants.image_read( ants.get_ants_data('r16') )
    >>> r64 = ants.image_read( ants.get_ants_data('r64') )
    >>> s16 = ants.kmeans_segmentation( r16, 3 )['segmentation']
    >>> s64 = ants.kmeans_segmentation( r64, 3 )['segmentation']
    >>> stats = ants.label_overlap_measures(s16, s64)
    """
    source_image_int = source_image.clone("unsigned int")
    target_image_int = target_image.clone("unsigned int")

    libfn = utils.get_lib_fn("labelOverlapMeasures%iD" % source_image_int.dimension)
    df = libfn(source_image_int.pointer, target_image_int.pointer)
    df = pd.DataFrame(df)
    df.Label[0] = "All"

    return df
