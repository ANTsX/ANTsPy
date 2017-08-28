
 

__all__ = ['label_clusters']

from .. import lib
from .process_args import _int_antsProcessArguments
from .threshold_image import threshold_image


def label_clusters(img, min_cluster_size=50, min_thresh=1e-6, max_thresh=1, fully_connected=False):
    """
    Label Clusters
    """
    dim = img.dimension
    clust = threshold_image(img, min_thresh, max_thresh)
    temp = int(fully_connected)
    args = [dim, clust, clust, min_cluster_size, temp]
    processed_args = _int_antsProcessArguments(args)
    lib.LabelClustersUniquely(processed_args)
    return clust