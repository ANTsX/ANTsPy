__all__ = ["crop_image_center",
           "pad_image_by_factor",
           "pad_or_crop_image_to_size",
           "crop_image_from_center_point"]

import ants
import numpy as np
import math

def crop_image_center(image,
                      crop_size):
    """
    Crop the center of an image.

    Arguments
    ---------
    image : ANTsImage
        Input image

    crop_size: n-D tuple
        Width, height, depth (if 3-D), and time (if 4-D) of crop region.

    Returns
    -------
    ANTs image.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> cropped_image = crop_image_center(image, crop_size=(64, 64))
    """

    image_size = np.array(image.shape)

    if len(image_size) != len(crop_size):
        raise ValueError("crop_size does not match image size.")

    if (np.asarray(crop_size) > np.asarray(image_size)).any():
        raise ValueError("A crop_size dimension is larger than image_size.")

    start_index = (np.floor(0.5 * (np.asarray(image_size) - np.asarray(crop_size)))).astype(int)
    end_index = start_index + np.asarray(crop_size).astype(int)

    cropped_image = ants.crop_indices(ants.image_clone(image) * 1, start_index, end_index)

    return(cropped_image)

def pad_image_by_factor(image,
                        factor):
    """
    Pad an image based on a factor.

    Pad image of size (x, y, z) to (x', y', z') where (x', y', z')
    is a divisible by a user-specified factor.

    Arguments
    ---------
    image : ANTsImage
        Input image

    factor: scalar or n-D tuple
        Padding factor

    Returns
    -------
    ANTs image.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> padded_image = pad_image_by_factor(image, factor=4)
    """

    factor_vector = factor
    if isinstance(factor, int):
        factor_vector = np.repeat(factor, image.dimension)

    if len(factor_vector) != image.dimension:
        raise ValueError("factor must be scalar or the length of the image dimension.")

    image_size = np.array(image.shape)
    delta_size = image_size % factor_vector

    padded_size = image_size
    for i in range(len(padded_size)):
        if delta_size[i] > 0:
            padded_size[i] = image_size[i] - delta_size[i] + factor_vector[i]

    padded_image = pad_or_crop_image_to_size(image, padded_size)

    return(padded_image)

def pad_or_crop_image_to_size(image,
                              size):
    """
    Pad or crop an image to a specified size

    Arguments
    ---------
    image : ANTsImage
        Input image

    size : tuple
        size of output image

    Returns
    -------
    A cropped or padded image

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> padded_image = pad_or_crop_image_to_size(image, (333, 333))
    """

    image_size = np.array(image.shape)

    delta = image_size - np.array(size)

    if np.any(delta < 0):
        pad_size = 2 * math.ceil(0.5 * abs(delta.min()))
        pad_shape = image_size + pad_size
        image = ants.pad_image(image, shape=pad_shape)

    cropped_image = crop_image_center(image, size)

    return(cropped_image)


def crop_image_from_center_point(image,
                                 center_point,
                                 patch_size):
    
    """
    Crop a patch from an image around a center point

    image: ANTsImage
        Input image

    center_point: tuple
        Physical space coordinates of center point.

    patch_size: tuple
        List defining patch size.

    """

    index = np.round(ants.transform_physical_point_to_index(image, center_point))
    image_shape = image.shape

    for k in range(len(image_shape)):
        if index[k] < 0:
            index[k] = 0
        if index[k] > (image_shape[k] - 1):
            index[k] = image_shape[k] - 1
    
    index_offset = index.copy()
    for k in range(len(patch_size)):
        index_offset[k] = index_offset[k] - np.round(0.5 * patch_size[k])

    domain_low = []
    domain_high = []
    for k in range(len(image_shape)):
        domain_low.append(int(index_offset[k] - 1))
        domain_high.append(int(index_offset[k] + 1))
    domain = ants.crop_indices(image, tuple(domain_low), tuple(domain_high) )

    patch_image = ants.make_image(patch_size)
    patch_image = ants.copy_image_info(domain, patch_image)

    return ants.resample_image_to_target(image, patch_image)
