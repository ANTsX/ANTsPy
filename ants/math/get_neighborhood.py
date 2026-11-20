
__all__ = ['get_neighborhood_in_mask',
           'get_neighborhood_at_voxel']

import numpy as np

import ants

from ants.internal import get_lib_fn, is_real_scalar
from ants.decorators import image_method

@image_method
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
    if not ants.is_image(image):
        raise ValueError('image must be ANTsImage type')
    if not ants.is_image(mask):
        raise ValueError('mask must be ANTsImage type')
    if isinstance(radius, np.ndarray):
        radius = radius.tolist()
    if is_real_scalar(radius):
        radius = [radius]*image.dimension

    if (not isinstance(radius, (tuple,list))) or (len(radius) != image.dimension):
        raise ValueError('radius must be tuple or list with length == image.dimension')

    boundary = 0
    if boundary_condition == 'image':
        boundary = 1
    elif boundary_condition == 'mean':
        boundary = 2

    libfn = get_lib_fn('getNeighborhoodMatrix%s' % image._libsuffix)
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

@image_method
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
        indices for neighborhood center. As in ANTsR, these are 1-based, and
        the returned indices are 1-based as well.

    kernel : tuple/list/ndarray
        either a collection of values for neighborhood radius (in voxels),
        one per image dimension, or a binary array with the same number of
        dimensions as the image and an odd size along each axis, specifying
        the shape of the neighborhood to extract

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
    if not ants.is_image(image):
        raise ValueError('image must be ANTsImage type')

    if isinstance(center, np.ndarray):
        center = center.tolist()
    if (not isinstance(center, (tuple,list))) or (len(center) != image.dimension):
        raise ValueError('center must be tuple or list with length == image.dimension')

    if isinstance(kernel, (tuple, list, np.ndarray)):
        kernel_array = np.asarray(kernel, dtype=float)
    else:
        kernel_array = None

    if kernel_array is not None and kernel_array.ndim == image.dimension:
        # binary neighborhood shape; the C++ side walks the neighborhood with
        # the first axis fastest, so flatten in Fortran order
        if any(s % 2 == 0 for s in kernel_array.shape):
            raise ValueError('kernel array must have an odd size along each axis')
        radius = [(s - 1) // 2 for s in kernel_array.shape]
        kernel = kernel_array.flatten(order='F').tolist()
    elif kernel_array is not None and kernel_array.ndim == 1 and len(kernel_array) == image.dimension:
        # neighborhood radius: extract the full (2*radius+1) hypercube
        if np.any(kernel_array < 0) or np.any(kernel_array != np.round(kernel_array)):
            raise ValueError('kernel radius values must be non-negative integers')
        radius = [int(r) for r in kernel_array]
        kernel = [1.0] * int(np.prod([2 * r + 1 for r in radius]))
    else:
        raise ValueError('kernel must be a radius with length == image.dimension '
                         'or an array with image.dimension dimensions')

    libfn = get_lib_fn('getNeighborhood%s' % image._libsuffix)
    retvals = libfn(image.pointer,
                    list(center),
                    list(kernel),
                    list(radius),
                    int(physical_coordinates))
    for k in retvals.keys():
        retvals[k] = np.asarray(retvals[k])
    return retvals


