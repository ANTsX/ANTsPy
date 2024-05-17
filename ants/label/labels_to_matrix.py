
__all__ = ['labels_to_matrix']

import numpy as np

from ..core import ants_image as iio


def labels_to_matrix(image, mask, target_labels=None, missing_val=np.nan):
    """
    Convert a labeled image to an n x m binary matrix where n = number of voxels
    and m = number of labels. Only includes values inside the provided mask while
    including background ( image == 0 ) for consistency with timeseries2matrix and
    other image to matrix operations.
    
    ANTsR function: `labels2matrix`

    Arguments
    ---------
    image : ANTsImage
        input label image

    mask : ANTsImage
        defines domain of interest

    target_labels : list/tuple
        defines target regions to be returned.  if the target label does not exist 
        in the input label image, then the matrix will contain a constant value 
        of missing_val (default None) in that row.

    missing_val : scalar
        value to use for missing label values

    Returns
    -------
    ndarray

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16')).resample_image((60,60),1,0)
    >>> mask = ants.get_mask(fi)
    >>> labs = ants.kmeans_segmentation(fi,3)['segmentation']
    >>> labmat = ants.labels_to_matrix(labs, mask)
    """
    if (not isinstance(image, iio.ANTsImage)) or (not isinstance(mask, iio.ANTsImage)):
        raise ValueError('image and mask must be ANTsImage types')

    vec = image[mask > 0]
    
    if target_labels is not None:
        the_labels = target_labels
    else:
        the_labels = np.sort(np.unique(vec))

    n_labels = len(the_labels)
    labels = np.zeros((n_labels, len(vec)))

    for i in range(n_labels):
        lab = float(the_labels[i])
        filler = (vec == lab).astype('float')
        if np.sum(filler) == 0:
            filler = np.asarray([np.nan]*len(vec))
        labels[i,:] = filler
    return labels
