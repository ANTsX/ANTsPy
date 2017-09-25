
__all__ = ['get_centroids']

import numpy as np

from . import label_clusters, label_stats

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
    >>> image = ants.image_read( ants.get_ants_data( "r16" ) )
    >>> image = ants.threshold_image( image, 90, 120 )
    >>> image = ants.label_clusters( image, 10 )
    >>> cents = ants.get_centroids( image  )
    """
    imagedim = image.dimension
    if clustparam > 0:
        mypoints = label_clusters(image, clustparam, max_thresh=1e15)
    if clustparam == 0:
        mypoints = image.clone()
    mypoints = label_stats(mypoints, mypoints)
    mypoints = mypoints[-1,:]
    x = mypoints[:,0]
    y = mypoints[:,1]
    
    if imagedim == 3:
        z = mypoints[:,2]
    else:
        z = np.zeros(mypoints.shape[0])

    if imagedim == 4:
        t = mypoints[:,4]
    else:
        t = np.zeros(mypoints.shape[0])

    centroids = np.hstack([x,y,z,t])
    return centroids

