
__all__ = ['image_to_cluster_images']

import numpy as np

from ..core import ants_image as iio
from .label_clusters import label_clusters

def image_to_cluster_images(img, min_cluster_size=50, min_thresh=1e-06, max_thresh=1):
    """
    Converts an image to several independent images.

    Produces a unique image for each connected 
    component 1 through N of size > min_cluster_size

    ANTsR function: `image2ClusterImages`

    Arguments
    ---------
    img : ANTsImage
        input image
    min_cluster_size : integer
        throw away clusters smaller than this value
    min_thresh : scalar
        threshold to a statistical map
    max_thresh : scalar
        threshold to a statistical map

    Returns
    -------
    list of ANTsImage types

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> img = ants.threshold_image(img, 1, 1e15)
    >>> img_cluster_list = ants.image_to_cluster_images(img)
    """
    if not isinstance(img, iio.ANTsImage):
        raise ValueError('img must be ANTsImage type')

    clust = label_clusters(img, min_cluster_size, min_thresh, max_thresh)
    labs = np.unique(clust[clust > 0])

    clustlist = []
    for i in range(len(labs)):
        labimg = img.clone()
        labimg[clust != labs[i]] = 0
        clustlist.append(labimg)
    return clustlist

