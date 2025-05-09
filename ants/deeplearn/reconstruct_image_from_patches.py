__all__ = ["reconstruct_image_from_patches"]

import ants
import numpy as np

def reconstruct_image_from_patches(patches,
                                   domain_image,
                                   stride_length=1,
                                   domain_image_is_mask=False):
    """
    Reconstruct image from a list of patches.

    Arguments
    ---------
    patches : list or array of patches
        List or array of patches defining an image.  Patches are assumed
        to have the same format as returned by extract_image_patches.

    domain_image : ANTs image
        Image or mask to define the geometric information of the
        reconstructed image.  If this is a mask image, the reconstruction will only
        use patches in the mask.

    stride_length:  integer or n-D tuple
        Defines the sequential patch overlap for max_number_of_patches = "all".
        Can be a image-dimensional vector or a scalar.

    domain_image_is_mask : boolean
        Boolean specifying whether the domain image is a
        mask used to limit the region of reconstruction from the patches.

    Returns
    -------
    An ANTs image.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> image_patches = extract_image_patches(image, patch_size=(16, 16), stride_length=4)
    >>> reconstructed_image = reconstruct_image_from_patches(image_patches, image, stride_length=4)
    """

    image_size = domain_image.shape
    dimensionality = domain_image.dimension

    if dimensionality != 2 and dimensionality != 3:
        raise ValueError("Unsupported dimensionality.")

    is_list_patches = isinstance(patches, list)

    patch_size = ()
    number_of_image_components = 1
    if is_list_patches:
        patch_dimension = patches[0].shape
        patch_size = patch_dimension[0:dimensionality]
        if len(patch_dimension) > dimensionality:
            number_of_image_components = patch_dimension[dimensionality]
    else:
        patch_dimension = patches.shape
        patch_size = patch_dimension[1:( 2 + dimensionality - 1 )]
        if len(patch_dimension) > dimensionality + 1:
            number_of_image_components = patch_dimension[dimensionality + 1]

    mid_patch_index = tuple(np.int_(np.subtract(np.round(0.5 * (np.array(patch_size))), 1)))

    image_array = np.zeros((*image_size, number_of_image_components))
    count_array = np.zeros((*image_size, number_of_image_components))

    stride_length_tuple = stride_length
    if isinstance(stride_length, int):
        stride_length_tuple = tuple(np.multiply(np.ones_like(patch_size), stride_length))
    elif len(stride_length) != dimensionality:
        raise ValueError("stride_length is not a scalar or vector of length dimensionality.")
    elif np.any(np.less(stride_length, 1)):
        raise ValueError("stride_length elements must be positive integers.")

    if domain_image_is_mask:
        mask_array = domain_image.numpy()
        mask_array[np.where(mask_array != 0)] = 1

    count = 0
    if dimensionality == 2:
        if np.all(np.equal(stride_length_tuple, 1)):
            for i in range(image_size[0] - patch_size[0] + 1):
                for j in range(image_size[1] - patch_size[1] + 1):
                    start_index = (i, j)
                    end_index = tuple(np.add(start_index, patch_size))

                    do_add = True
                    if domain_image_is_mask:
                        if mask_array[start_index[0]:end_index[0],
                                      start_index[1]:end_index[1]][mid_patch_index[0],
                                                                   mid_patch_index[1]] == 0:
                            do_add = False

                    if do_add:
                        patch = np.empty((1, 1))
                        if is_list_patches:
                            patch = patches[count]
                        else:
                            if number_of_image_components == 1:
                                patch = patches[count, :, :]
                            else:
                                patch = patches[count, :, :, :]

                        if number_of_image_components == 1:
                            patch = np.expand_dims(patch, axis = 2)

                        image_array[start_index[0]:end_index[0],
                                    start_index[1]:end_index[1], :] += patch
                        count_array[start_index[0]:end_index[0],
                                    start_index[1]:end_index[1], :] += 1
                        count += 1

            image_array[count_array>0] = image_array[count_array>0] / count_array[count_array>0]

            if not domain_image_is_mask:
                for i in range(image_size[0]):
                    for j in range(image_size[1]):
                        factor = (min(i + 1, patch_size[0], image_size[0] - i) *
                                  min(j + 1, patch_size[1], image_size[1] - j))

                        image_array[i, j, :] /= factor
        else:
            for i in range(0, image_size[0] - patch_size[0] + 1, stride_length_tuple[0]):
                for j in range(0, image_size[1] - patch_size[1] + 1, stride_length_tuple[1]):
                    start_index = (i, j)
                    end_index = tuple(np.add(start_index, patch_size))

                    patch = np.empty((1, 1))
                    if is_list_patches:
                        patch = patches[count]
                    else:
                        if number_of_image_components == 1:
                            patch = patches[count, :, :]
                        else:
                            patch = patches[count, :, :, :]

                    if number_of_image_components == 1:
                        patch = np.expand_dims(patch, axis=2)

                    image_array[start_index[0]:end_index[0],
                                start_index[1]:end_index[1], :] += patch
                    count_array[start_index[0]:end_index[0],
                                start_index[1]:end_index[1], 0] += np.ones(patch_size)
                    count += 1

            count_array[np.where(count_array == 0)] = 1
            for i in range(number_of_image_components):
                image_array[:, :, i] /= count_array[:,:,0]

    else:

        if np.all(np.equal(stride_length_tuple, 1)):
            for i in range(image_size[0] - patch_size[0] + 1):
                for j in range(image_size[1] - patch_size[1] + 1):
                    for k in range(image_size[2] - patch_size[2] + 1):
                        start_index = (i, j, k)
                        end_index = tuple(np.add(start_index, patch_size))

                        do_add = True
                        if domain_image_is_mask:
                            if mask_array[start_index[0]:end_index[0],
                                          start_index[1]:end_index[1],
                                          start_index[2]:end_index[2]][mid_patch_index[0],
                                                                       mid_patch_index[1],
                                                                       mid_patch_index[2]] == 0:
                                do_add = False

                        if do_add:
                            patch = np.empty((1, 1, 1))
                            if is_list_patches:
                                patch = patches[count]
                            else:
                                if number_of_image_components == 1:
                                    patch = patches[count, :, :, :]
                                else:
                                    patch = patches[count, :, :, :, :]

                            if number_of_image_components == 1:
                                patch = np.expand_dims(patch, axis=3)

                            image_array[start_index[0]:end_index[0],
                                        start_index[1]:end_index[1],
                                        start_index[2]:end_index[2], :] += patch
                            count_array[start_index[0]:end_index[0],
                                        start_index[1]:end_index[1],
                                        start_index[2]:end_index[2], :] += 1
                            count += 1

            image_array[count_array>0] = image_array[count_array>0] / count_array[count_array>0]


            if not domain_image_is_mask:
                for i in range(image_size[0] + 1):
                    for j in range(image_size[1] + 1):
                        for k in range(image_size[2] + 1):
                            factor = (min(i + 1, patch_size[0], image_size[0] - i) *
                                      min(j + 1, patch_size[1], image_size[1] - j) *
                                      min(k + 1, patch_size[2], image_size[2] - k))

                            image_array[i, j, k, :] /= factor
        else:
            for i in range(0, image_size[0] - patch_size[0] + 1, stride_length_tuple[0]):
                for j in range(0, image_size[1] - patch_size[1] + 1, stride_length_tuple[1]):
                    for k in range(0, image_size[2] - patch_size[2] + 1, stride_length_tuple[2]):
                        start_index = (i, j, k)
                        end_index = tuple(np.add(start_index, patch_size))

                        patch = np.empty((1, 1, 1))
                        if is_list_patches:
                            patch = patches[count]
                        else:
                            if number_of_image_components == 1:
                                patch = patches[count, :, :, :]
                            else:
                                patch = patches[count, :, :, :, :]

                        if number_of_image_components == 1:
                            patch = np.expand_dims(patch, axis=3)

                        image_array[start_index[0]:end_index[0],
                                    start_index[1]:end_index[1],
                                    start_index[2]:end_index[2], :] += patch
                        count_array[start_index[0]:end_index[0],
                                    start_index[1]:end_index[1],
                                    start_index[2]:end_index[2], 0] += np.ones(patch_size)
                        count += 1

            count_array[np.where(count_array == 0)] = 1
            for i in range(number_of_image_components):
                image_array[:, :, :, i] /= count_array[:,:,:,0]

    if dimensionality == 2:
        image_array = np.transpose(image_array, [2, 0, 1])
    else:
        image_array = np.transpose(image_array, [3, 0, 1, 2])

    image_array = np.squeeze(image_array)

    reconstructed_image = ants.from_numpy(image_array,
                                          origin=domain_image.origin,
                                          spacing=domain_image.spacing,
                                          direction=domain_image.direction,
                                          has_components=(number_of_image_components != 1))

    return(reconstructed_image)
