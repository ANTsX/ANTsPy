

__all__ = ['make_points_image']

import math
import numpy as np

from ..core import ants_transform as tio

def make_points_image(pts, mask, radius=5):
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
        input powers points

    mask : ANTsImage
        mask defining target space

    radius : integer
        radius for the points
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> import pandas as pd
    >>> mni = ants.image_read(ants.get_data('mni')).get_mask()
    >>> powers_pts = pd.read_csv(ants.get_data('powers_areal_mni_itk'))
    >>> powers_labels = ants.make_points_image(powers_pts.iloc[:,:3].values, mni, radius=3)
    """
    powers_lblimg = mask * 0
    npts = len(pts)
    rad = radius
    n = math.ceil(rad/np.array(mask.spacing))
    dim = mask.dimension
    if pts.shape[1] < dim:
        raise ValueError('points dimensionality should match that of images')

    for r in range(npts):
        pt = pts[r,:dim]
        idx = tio.transform_physical_point_to_index(mask, pt)
        for i in np.arange(-n[0],n[0],step=0.5):
            for j in np.arange(-n[1],n[1],step=0.5):
                if (dim == 3):
                    for k in np.arange(-n[2],n[2],step=0.5):
                        local = idx + np.array([i,j,k])
                        localpt = tio.transform_index_to_physical_point(mask, local)
                        dist = np.sqrt(np.sum((localpt-pt)*(localpt-pt)))
                        in_image = (np.prod(idx <= mask.dimension)==1) and (len(np.where(idx<1))==0)
                        if ( (dist <= rad) and (in_image == True) ):
                            if powers_lblimg[local[0],local[1],local[2]] < 0.5:
                                powers_lblimg[local[0],local[1],local[2]] = r
                elif (dim == 2):
                    local = idx + np.array([i,j])
                    localpt = tio.transform_index_to_physical_point(mask, local)
                    dist = np.sqrt(np.sum((localpt-pt)*(localpt-pt)))
                    in_image = (np.prod(idx<=mask.dimension)==1) and (len(np.where(idx<1))==0)
                    if ((dist <= rad) and (in_image == True)):
                        if powers_lblimg[local[0],local[1]] < 0.5:
                            powers_lblimg[local[0],local[1]] = r
    return powers_lblimg

