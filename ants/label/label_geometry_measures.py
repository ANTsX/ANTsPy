
__all__ = ['label_geometry_measures']

from tempfile import mktemp
import pandas as pd
import numpy as np

from .. import utils


def label_geometry_measures(label_image, intensity_image=None):
    """
    Wrapper for the ANTs funtion labelGeometryMeasures
    
    ANTsR function: `labelGeometryMeasures`

    Arguments
    ---------
    label_image : ANTsImage
        image on which to compute geometry
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

    veccer = [label_image.dimension, label_image, intensity_image, outcsv]
    veccer_processed = utils._int_antsProcessArguments(veccer)
    libfn = utils.get_lib_fn('LabelGeometryMeasures')
    pp = libfn(veccer_processed)
    pp = pd.read_csv(outcsv)
    pp['Label'] = np.sort(np.unique(label_image[label_image>0])).astype('int')
    pp_cols = pp.columns.values
    pp_cols[1] = 'VolumeInMillimeters'
    pp.columns = pp_cols
    spc = np.prod(label_image.spacing)
    pp['VolumeInMillimeters'] = pp['VolumeInMillimeters']*spc
    return pp



