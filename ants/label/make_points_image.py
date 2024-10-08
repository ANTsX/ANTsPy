

__all__ = ['make_points_image']

import math
import numpy as np

import ants

def make_points_image(pts, target, radius=5):
    """
    Create label image from physical space points

    Creates spherical points in the coordinate space of the target image based
    on the n-dimensional matrix of points that the user supplies. The image
    defines the dimensionality of the data so if the input image is 3D then
    the input points should be 2D or 3D.

    ANTsR function: `makePointsImage`

    Arguments
    ---------
    pts : numpy.ndarray
        input points

    target : ANTsImage
        Image defining target space

    radius : integer
        radius for the points

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> import pandas as pd
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> powers_pts = pd.read_csv(ants.get_data('powers_mni_itk'))
    >>> powers_labels = ants.make_points_image(powers_pts.iloc[:,:3].values, mni, radius=3)
    """
    lblimg = target * 0
    npts = len(pts)
    dim = target.dimension
    if pts.shape[1] != dim:
        raise ValueError('points dimensionality should match that of images')

    for r in range(npts):
        pt = pts[r,:]
        idx = ants.transform_physical_point_to_index(target, pt.tolist() ).astype(int)
        in_image=True
        for kk in range(target.dimension):
            in_image = in_image and idx[kk] >= 0 and idx[kk] < target.shape[kk]
        if ( in_image == True ):
            if (dim == 3):
                lblimg[idx[0],idx[1],idx[2]] = r + 1
            elif (dim == 2):
                lblimg[idx[0],idx[1]] = r + 1
    return ants.morphology( lblimg, 'dilate', radius, 'grayscale' )
