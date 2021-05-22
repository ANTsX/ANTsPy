
__all__ = ['fuzzy_spatial_cmeans_segmentation']

import numpy as np

from .. import core
from .. import utils

def fuzzy_spatial_cmeans_segmentation(image,
                                      mask=None,
                                      number_of_clusters=4,
                                      m=2,
                                      p=1,
                                      q=1,
                                      radius=2,
                                      max_number_of_iterations=20,
                                      convergence_threshold=0.02,
                                      verbose=False):
    """
    Fuzzy spatial c-means for image segmentation.

    Image segmentation using fuzzy spatial c-means as described in

    Chuang et al., Fuzzy c-means clustering with spatial information for image
    segmentation.  CMIG: 30:9-15, 2006.

    Arguments
    ---------
    image : ANTsImage
        Input image.

    mask : ANTsImage
        Optional mask image.

    number_of_clusters : integer
        Number of segmentation clusters.

    m : float
        Fuzziness parameter (default=2).

    p : float
        Membership importance parameter (default=1).

    q : float
        Spatial constraint importance parameter (default=1).
        q = 0 is equivalent to conventional fuzzy c-means.

    radius : integer or tuple
        Neighborhood radius (scalar or array) for spatial constraint.

    max_number_of_iterations : integer
        Iteration limit (default=20).

    convergence_threshold : float
        Convergence between iterations is measured using the Dice coefficient
        (default=0.02).

    varbose : boolean
        Print progress.

    Returns
    -------
    dictionary containing ANTsImage and probability images

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> mask = ants.get_mask(image)
    >>> fuzzy = ants.fuzzy_spatial_cmeans_segmentation(image, mask, number_of_clusters=3)
    """

    if mask is None:
        mask = core.image_clone(image) * 0 + 1

    x = image[mask != 0]

    # This is a hack because the order of the voxels is not the same for the following
    # two operations
    #        x = image[mask != 0]
    #        x = ants.get_neighborhood_in_mask(image, mask)

    mask_perm = core.from_numpy(np.transpose(mask.numpy()))

    v = np.linspace(0, 1, num=(number_of_clusters + 2))[1:(number_of_clusters+1)]
    v = v * (x.max() - x.min()) + x.min()
    cc = len(v)

    if verbose == True:
        print("Initial cluster centers: ", v)

    xx = np.zeros((cc, len(x)))
    for i in range(cc):
        xx[i,:] = x

    if isinstance(radius, int):
        radius = tuple(np.zeros((image.dimension,), dtype=int) + radius)

    segmentation = core.image_clone(image) * 0
    probability_images = None

    np.seterr(divide='ignore', invalid='ignore')

    iter = 0
    dice_value = 0
    while iter < max_number_of_iterations and dice_value < 1.0 - convergence_threshold:

        # Update membership values

        xv = np.zeros((cc, len(x)))
        for k in range(cc):
            xv[k,:] = abs(x - v[k])

        u = np.zeros((xv.shape[0], xv.shape[1]))
        for i in range(cc):
            n = xv[i,:]

            d = n * 0
            for k in range(cc):
                d += (n / xv[k,:]) ** (2 / (m - 1))

            u[i,:] = 1 / d
        u = np.nan_to_num(u, nan=1.0)

        # Update cluster centers

        v = np.nansum((u ** m) * xx, axis=1) / np.nansum((u ** m), axis=1)

        if verbose == True:
            print("Updated cluster centers: ", v)

        # Spatial function

        h = np.zeros((u.shape[0], u.shape[1]))
        for i in range(cc):
            u_image = core.image_clone(image) * 0
            u_image[mask != 0] = u[i,:]
            u_image_perm = core.from_numpy(np.transpose(u_image.numpy()))
            u_neighborhoods = utils.get_neighborhood_in_mask(u_image_perm, mask_perm, radius)
            h[i,:] = np.nansum(u_neighborhoods, axis=0)

        # u prime

        d = np.zeros((u.shape[1],))
        for k in range(cc):
            d += (u[k,:] ** p) * (h[k,:] ** q)

        probability_images = []
        uprime = np.zeros((u.shape[0], u.shape[1]))
        for i in range(cc):
            uprime[i,:] = ((u[i,:] ** p) * (h[i,:] ** q)) / d
            uprime_image = core.image_clone(image) * 0
            uprime_image[mask != 0] = uprime[i,:]
            probability_images.append(uprime_image)

        tmp_segmentation = core.image_clone(image) * 0
        tmp_segmentation[mask != 0] = np.argmax(uprime, axis=0) + 1

        dice_value = utils.label_overlap_measures(segmentation, tmp_segmentation)['MeanOverlap'][0]
        iter = iter + 1

        if verbose == True:
            print("Iteration ", iter, " (out of ", max_number_of_iterations, "):  ",
                "Dice overlap = ", dice_value, sep = "")

        segmentation = tmp_segmentation

    return_dict = {'segmentation_image' : segmentation,
                   'probability_images' : probability_images}
    return(return_dict)