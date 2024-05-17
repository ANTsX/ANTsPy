
__all__ = [
    "matrix_to_images",
    "images_from_matrix",
    "image_list_to_matrix",
    "images_to_matrix",
    "matrix_from_images",
    "timeseries_to_matrix",
    "matrix_to_timeseries"
]

import os
import json
import numpy as np
import warnings


from . import ants_image as iio
from .. import utils, core
from .. import registration as reg


def matrix_to_timeseries(image, matrix, mask=None):
    """
    converts a matrix to a ND image.

    ANTsR function: `matrix2timeseries`

    Arguments
    ---------

    image: reference ND image

    matrix: matrix to convert to image

    mask: mask image defining voxels of interest


    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.make_image( (10,10,10,5 ) )
    >>> mask = ants.ndimage_to_list( img )[0] * 0
    >>> mask[ 4:8, 4:8, 4:8 ] = 1
    >>> mat = ants.timeseries_to_matrix( img, mask = mask )
    >>> img2 = ants.matrix_to_timeseries( img,  mat, mask)
    """

    if mask is None:
        mask = temp[0] * 0 + 1
    temp = matrix_to_images(matrix, mask)
    newImage = utils.list_to_ndimage(image, temp)
    iio.copy_image_info(image, newImage)
    return newImage


def matrix_to_images(data_matrix, mask):
    """
    Unmasks rows of a matrix and writes as images

    ANTsR function: `matrixToImages`

    Arguments
    ---------
    data_matrix : numpy.ndarray
        each row corresponds to an image
        array should have number of columns equal to non-zero voxels in the mask

    mask : ANTsImage
        image containing a binary mask. Rows of the matrix are
        unmasked and written as images. The mask defines the output image space

    Returns
    -------
    list of ANTsImage types

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> msk = ants.get_mask( img )
    >>> img2 = ants.image_read(ants.get_ants_data('r16'))
    >>> img3 = ants.image_read(ants.get_ants_data('r16'))
    >>> mat = ants.image_list_to_matrix([img,img2,img3], msk )
    >>> ilist = ants.matrix_to_images( mat, msk )
    """

    if data_matrix.ndim > 2:
        data_matrix = data_matrix.reshape(data_matrix.shape[0], -1)

    numimages = len(data_matrix)
    numVoxelsInMatrix = data_matrix.shape[1]
    numVoxelsInMask = (mask >= 0.5).sum()
    if numVoxelsInMask != numVoxelsInMatrix:
        raise ValueError(
            "Num masked voxels %i must match data matrix %i"
            % (numVoxelsInMask, numVoxelsInMatrix)
        )

    imagelist = []
    for i in range(numimages):
        img = mask.clone()
        img[mask >= 0.5] = data_matrix[i, :]
        imagelist.append(img)
    return imagelist


images_from_matrix = matrix_to_images


def images_to_matrix(image_list, mask=None, sigma=None, epsilon=0.5):
    """
    Read images into rows of a matrix, given a mask - much faster for
    large datasets as it is based on C++ implementations.

    ANTsR function: `imagesToMatrix`

    Arguments
    ---------
    image_list : list of ANTsImage types
        images to convert to ndarray

    mask : ANTsImage (optional)
        image containing binary mask. voxels in the mask are placed in the matrix

    sigma : scaler (optional)
        smoothing factor

    epsilon : scalar
        threshold for mask

    Returns
    -------
    ndarray
        array with a row for each image
        shape = (N_IMAGES, N_VOXELS)

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> img2 = ants.image_read(ants.get_ants_data('r16'))
    >>> img3 = ants.image_read(ants.get_ants_data('r16'))
    >>> mat = ants.image_list_to_matrix([img,img2,img3])
    """

    def listfunc(x):
        if np.sum(np.array(x.shape) - np.array(mask.shape)) != 0:
            x = reg.resample_image_to_target(x, mask, 2)
        return x[mask]

    if mask is None:
        mask = utils.get_mask(image_list[0])

    num_images = len(image_list)
    mask_arr = mask.numpy() >= epsilon
    num_voxels = np.sum(mask_arr)

    data_matrix = np.empty((num_images, num_voxels))
    do_smooth = sigma is not None
    for i, img in enumerate(image_list):
        if do_smooth:
            data_matrix[i, :] = listfunc(
                utils.smooth_image(img, sigma, sigma_in_physical_coordinates=True)
            )
        else:
            data_matrix[i, :] = listfunc(img)
    return data_matrix


image_list_to_matrix = images_to_matrix
matrix_from_images = images_to_matrix


def timeseries_to_matrix(image, mask=None):
    """
    Convert a timeseries image into a matrix.

    ANTsR function: `timeseries2matrix`

    Arguments
    ---------
    image : image whose slices we convert to a matrix. E.g. a 3D image of size
           x by y by z will convert to a z by x*y sized matrix

    mask : ANTsImage (optional)
        image containing binary mask. voxels in the mask are placed in the matrix

    Returns
    -------
    ndarray
        array with a row for each image
        shape = (N_IMAGES, N_VOXELS)

    Example
    -------
    >>> import ants
    >>> img = ants.make_image( (10,10,10,5 ) )
    >>> mat = ants.timeseries_to_matrix( img )
    """
    temp = utils.ndimage_to_list(image)
    if mask is None:
        mask = temp[0] * 0 + 1
    return image_list_to_matrix(temp, mask)
