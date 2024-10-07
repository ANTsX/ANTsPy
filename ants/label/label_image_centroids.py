

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
        image of integer labels

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

        `vertices` : pd.DataFrame
            coordinates of label centroids

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> image = ants.from_numpy(np.asarray([[[0,2],[1,3]],[[4,6],[5,7]]]).astype('float32'))
    >>> labels = ants.label_image_centroids(image)
    """
    d = image.shape
    if len(d) != 3:
        raise ValueError('image must be 3 dimensions')

    xcoords = np.asarray(np.arange(d[0]).tolist()*(d[1]*d[2]))
    ycoords = np.asarray(np.repeat(np.arange(d[1]),d[0]).tolist()*d[2])
    zcoords = np.asarray(np.repeat(np.arange(d[1]), d[0]*d[2]))

    labels = image.numpy()
    mylabels = np.sort(np.unique(labels[labels > 0])).astype('int')
    n_labels = len(mylabels)
    xc = np.zeros(n_labels)
    yc = np.zeros(n_labels)
    zc = np.zeros(n_labels)

    if convex:
        for lab_idx, label_intensity in enumerate(mylabels):
            idx = (labels == label_intensity).flatten()
            xc[lab_idx] = np.mean(xcoords[idx])
            yc[lab_idx] = np.mean(ycoords[idx])
            zc[lab_idx] = np.mean(zcoords[idx])
    else:
        for lab_idx, label_intensity in enumerate(mylabels):
            idx = (labels == label_intensity).flatten()
            xci = xcoords[idx]
            yci = ycoords[idx]
            zci = zcoords[idx]
            dist = np.zeros(len(xci))

            for j in range(len(xci)):
                dist[j] = np.mean(np.sqrt((xci[j] - xci)**2 + (yci[j] - yci)**2 + (zci[j] - zci)**2))

            mid = np.where(dist==np.min(dist))
            xc[lab_idx] = xci[mid]
            yc[lab_idx] = yci[mid]
            zc[lab_idx] = zci[mid]

    centroids = np.vstack([xc,yc,zc]).T

    #if physical:
    #    centroids = ants.transform_index_to_physical_point(image, centroids)

    return {
        'labels': mylabels,
        'vertices': centroids
    }




