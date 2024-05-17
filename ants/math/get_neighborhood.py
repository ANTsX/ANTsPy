
__all__ = ['get_neighborhood_in_mask',
            'get_neighborhood_at_voxel']

import numpy as np

from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2


def get_neighborhood_in_mask(image, mask, radius, physical_coordinates=False,
                            boundary_condition=None, spatial_info=False, get_gradient=False):
    """
    Get neighborhoods for voxels within mask.
    
    This converts a scalar image to a matrix with rows that contain neighbors 
    around a center voxel
    
    ANTsR function: `getNeighborhoodInMask`

    Arguments
    ---------
    image : ANTsImage
        image to get values from
    
    mask : ANTsImage
        image indicating which voxels to examine. Each voxel > 0 will be used as the 
        center of a neighborhood
    
    radius : tuple/list
        array of values for neighborhood radius (in voxels)
    
    physical_coordinates : boolean
        whether voxel indices and offsets should be in voxel or physical coordinates
    
    boundary_condition : string (optional)
        how to handle voxels in a neighborhood, but not in the mask.
            None : fill values with `NaN`
            `image` : use image value, even if not in mask
            `mean` : use mean of all non-NaN values for that neighborhood
    
    spatial_info : boolean
        whether voxel locations and neighborhood offsets should be returned along with pixel values.
    
    get_gradient : boolean
        whether a matrix of gradients (at the center voxel) should be returned in 
        addition to the value matrix (WIP)

    Returns
    -------
    if spatial_info is False:
        if get_gradient is False:
            ndarray
                an array of pixel values where the number of rows is the size of the 
                neighborhood and there is a column for each voxel

        else if get_gradient is True:
            dictionary w/ following key-value pairs:
                values : ndarray
                    array of pixel values where the number of rows is the size of the 
                    neighborhood and there is a column for each voxel.

                gradients : ndarray
                    array providing the gradients at the center voxel of each 
                    neighborhood
        
    else if spatial_info is True:
        dictionary w/ following key-value pairs:
            values : ndarray
                array of pixel values where the number of rows is the size of the 
                neighborhood and there is a column for each voxel.

            indices : ndarray
                array provinding the center coordinates for each neighborhood

            offsets : ndarray
                array providing the offsets from center for each voxel in a neighborhood

    Example
    -------
    >>> import ants
    >>> r16 = ants.image_read(ants.get_ants_data('r16'))
    >>> mask = ants.get_mask(r16)
    >>> mat = ants.get_neighborhood_in_mask(r16, mask, radius=(2,2))
    """
    if not isinstance(image, iio.ANTsImage):
        raise ValueError('image must be ANTsImage type')
    if not isinstance(mask, iio.ANTsImage):
        raise ValueError('mask must be ANTsImage type')
    if isinstance(radius, (int, float)):
        radius = [radius]*image.dimension
    if (not isinstance(radius, (tuple,list))) or (len(radius) != image.dimension):
        raise ValueError('radius must be tuple or list with length == image.dimension')

    boundary = 0
    if boundary_condition == 'image':
        boundary = 1
    elif boundary_condition == 'mean':
        boundary = 2
    
    libfn = utils.get_lib_fn('getNeighborhoodMatrix%s' % image._libsuffix)
    retvals = libfn(image.pointer, 
                    mask.pointer, 
                    list(radius),
                    int(physical_coordinates), 
                    int(boundary),
                    int(spatial_info),
                    int(get_gradient))

    if not spatial_info:
        if get_gradient:
            retvals['values'] = np.asarray(retvals['values'])
            retvals['gradients'] = np.asarray(retvals['gradients'])
        else:
            retvals = np.asarray(retvals['matrix'])
    else:
        retvals['values'] = np.asarray(retvals['values'])
        retvals['indices'] = np.asarray(retvals['indices'])
        retvals['offsets'] = np.asarray(retvals['offsets'])

    return retvals


def get_neighborhood_at_voxel(image, center, kernel, physical_coordinates=False):
    """
    Get a hypercube neighborhood at a voxel. Get the values in a local 
    neighborhood of an image.
    
    ANTsR function: `getNeighborhoodAtVoxel`

    Arguments
    ---------
    image : ANTsImage
        image to get values from.
    
    center : tuple/list
        indices for neighborhood center
    
    kernel : tuple/list
        either a collection of values for neighborhood radius (in voxels) or 
        a binary collection of the same dimension as the image, specifying the shape of the neighborhood to extract
    
    physical_coordinates : boolean
        whether voxel indices and offsets should be in voxel 
        or physical coordinates

    Returns
    -------
    dictionary w/ following key-value pairs:
        values : ndarray
            array of neighborhood values at the voxel

        indices : ndarray
            matrix providing the coordinates for each value

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> center = (2,2)
    >>> radius = (3,3)
    >>> retval = ants.get_neighborhood_at_voxel(img, center, radius)
    """
    if not isinstance(image, iio.ANTsImage):
        raise ValueError('image must be ANTsImage type')

    if (not isinstance(center, (tuple,list))) or (len(center) != image.dimension):
        raise ValueError('center must be tuple or list with length == image.dimension')

    if (not isinstance(kernel, (tuple,list))) or (len(kernel) != image.dimension):
        raise ValueError('kernel must be tuple or list with length == image.dimension')

    radius = [int((k-1)/2) for k in kernel]

    libfn = utils.get_lib_fn('getNeighborhood%s' % image._libsuffix)
    retvals = libfn(image.pointer, 
                    list(center), 
                    list(kernel), 
                    list(radius), 
                    int(physical_coordinates))
    for k in retvals.keys():
        retvals[k] = np.asarray(retvals[k])
    return retvals


