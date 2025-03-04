
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
        label_image_int = label_image.clone('unsigned int')
        if not np.all(label_image.numpy() == label_image_int.numpy()):
            raise ValueError('Input label values must be representable as uint32.')
        label_image = label_image.clone('unsigned int')
    veccer = [label_image.dimension, label_image_int, intensity_image, outcsv]
    veccer_processed = process_arguments(veccer)
    libfn = get_lib_fn('LabelGeometryMeasures')
    pp = libfn(veccer_processed)
    pp = pd.read_csv(outcsv)
    if 'VolumeInVoxels' in pp.columns and not 'VolumeInMillimeters' in pp.columns:
        spc = np.prod(label_image.spacing)
        pp['VolumeInMillimeters'] = pp['VolumeInVoxels'] * spc
    # Ensure that the label column is of integer type - if there is any NaN, it will be float
    # Something has gone seriously wrong if the labels are not interpreted as integers
    if not np.issubdtype(pp['Label'].dtype, np.integer):
        raise ValueError('Label column not integer type, label values may be invalid')
    return pp
