
__all__ = ['label_geometry_measures']

from tempfile import mktemp
import pandas as pd
import numpy as np

from ants.internal import get_lib_fn, process_arguments
from ants.decorators import image_method

@image_method
def label_geometry_measures(label_image, intensity_image=None):
    """
    Wrapper for the ANTs funtion LabelGeometryMeasures

    ANTsR function: `labelGeometryMeasures`

    Arguments
    ---------
    label_image : ANTsImage
        image on which to compute geometry. Labels must be representable as uint32.
    intensity_image : ANTsImage (optional)
        image with intensity values

    Returns
    -------
    pandas.DataFrame

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16') )
    >>> seg = ants.kmeans_segmentation( fi, 3 )['segmentation']
    >>> geom = ants.label_geometry_measures(seg,fi)
    """
    if intensity_image is None:
        intensity_image = label_image.clone()

    outcsv = mktemp(suffix='.csv')
    # Library function requires unsigned int labels
    if label_image.pixeltype != 'unsigned int':
        if label_image.max() > np.iinfo(np.uint32).max:
            raise ValueError('Labels must be representable as uint32')
        label_image = label_image.clone('unsigned int')
    veccer = [label_image.dimension, label_image, intensity_image, outcsv]
    veccer_processed = process_arguments(veccer)
    libfn = get_lib_fn('LabelGeometryMeasures')
    pp = libfn(veccer_processed)
    pp = pd.read_csv(outcsv)
    return pp



