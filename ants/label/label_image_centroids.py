

__all__ = ['label_image_centroids']

import numpy as np

from ants.decorators import image_method


@image_method
def label_image_centroids(image, physical=False, convex=True, verbose=False):
    """
    Converts a label image to coordinates summarizing their positions

    ANTsR function: `labelImageCentroids`

    Arguments
    ---------
    image : ANTsImage
        image of integer labels (2D, 3D or 4D)

    physical : boolean
        whether you want physical space coordinates or not

    convex : boolean
        if True, return centroid
        if False return point with min average distance to other points with same label

    Returns
    -------
    dictionary w/ following key-value pairs:
        `labels` : 1D-ndarray
            array of label values

        `vertices` : ndarray
            coordinates of label centroids, one row per label, in voxel
            index order (matching ``image.numpy()``), or in physical space
            if ``physical=True``

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> image = ants.from_numpy(np.asarray([[[0,2],[1,3]],[[4,6],[5,7]]]).astype('float32'))
    >>> labels = ants.label_image_centroids(image)
    """
    labels = image.numpy()
    mylabels = np.sort(np.unique(labels[labels > 0])).astype('int')
    n_labels = len(mylabels)

    centroids = np.zeros((n_labels, image.dimension))
    for lab_idx, label_intensity in enumerate(mylabels):
        # voxel indices, in the same (x, y, z) order as image.numpy()
        coords = np.argwhere(labels == label_intensity).astype('float64')
        if convex:
            centroids[lab_idx] = coords.mean(axis=0)
        else:
            dist = np.zeros(len(coords))
            for j in range(len(coords)):
                dist[j] = np.mean(np.sqrt(np.sum((coords[j] - coords)**2, axis=1)))
            centroids[lab_idx] = coords[np.argmin(dist)]

    if physical and n_labels > 0:
        spacing = np.asarray(image.spacing)
        origin = np.asarray(image.origin)
        direction = np.asarray(image.direction)
        centroids = origin + (centroids * spacing) @ direction.T

    return {
        'labels': mylabels,
        'vertices': centroids
    }
