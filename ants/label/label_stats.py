__all__ = ["label_stats"]

import numpy as np
import pandas as pd

from ants.internal import get_lib_fn
from ants.decorators import image_method

@image_method
def label_stats(image, label_image):
    """
    Get label statistics from an image. The labels must be representable as uint32.

    ANTsR function: `labelStats`

    Arguments
    ---------
    image : ANTsImage
        Image from which statistics will be calculated

    label_image : ANTsImage
        Label image

    Returns
    -------
    pandas.DataFrame

    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') , 2 )
    >>> image = ants.resample_image( image, (64,64), 1, 0 )
    >>> mask = ants.get_mask(image)
    >>> segs1 = ants.kmeans_segmentation( image, 3 )
    >>> stats = ants.label_stats(image, segs1['segmentation'])
    """
    image_float = image.clone("float")

    label_image_int = label_image.clone('unsigned int')

    if label_image.pixeltype != 'unsigned int':
        if not np.all(label_image.numpy() == label_image_int.numpy()):
            raise ValueError('Input label values must be representable as uint32.')

    libfn = get_lib_fn("labelStats%iD" % image.dimension)
    df = libfn(image_float.pointer, label_image_int.pointer)
    df = pd.DataFrame(df)
    df.sort_values(by=["LabelValue"], inplace=True)
    df = df.reset_index(drop=True)
    return df
