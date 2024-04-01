__all__ = ["get_centroids"]

import numpy as np

from .label_clusters import label_clusters
from .label_stats import label_stats


def get_centroids(image, clustparam=0):
    """
    Reduces a variate/statistical/network image to a set of centroids
    describing the center of each stand-alone non-zero component in the image

    ANTsR function: `getCentroids`

    Arguments
    ---------
    image : ANTsImage
        image from which centroids will be calculated

    clustparam : integer
        look at regions greater than or equal to this size

    Returns
    -------
    ndarray

    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data( "r16" ) )
    >>> image = ants.threshold_image( image, 90, 120 )
    >>> image = ants.label_clusters( image, 10 )
    >>> cents = ants.get_centroids( image )
    """
    imagedim = image.dimension
    if clustparam > 0:
        mypoints = label_clusters(image, clustparam, max_thresh=1e15)
    if clustparam == 0:
        mypoints = image.clone()
    mypoints = label_stats(mypoints, mypoints)
    nonzero = mypoints[["LabelValue"]] > 0
    mypoints = mypoints[nonzero["LabelValue"]]
    mypoints = mypoints.iloc[:, :]
    x = mypoints.x
    y = mypoints.y

    if imagedim == 3:
        z = mypoints.z
    else:
        z = np.zeros(mypoints.shape[0])

    if imagedim == 4:
        t = mypoints.t
    else:
        t = np.zeros(mypoints.shape[0])

    centroids = np.stack([x, y, z, t]).T
    return centroids
