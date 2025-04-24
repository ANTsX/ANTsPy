__all__ = ["extract_image_patches"]

import numpy as np
import random

def extract_image_patches(image,
                          patch_size,
                          max_number_of_patches='all',
                          stride_length=1,
                          mask_image=None,
                          random_seed=None,
                          return_as_array=False,
                          randomize=True):
    """
    Extract 2-D or 3-D image patches.

    Arguments
    ---------
    image : ANTsImage
        Input image with one or more components.

    patch_size: n-D tuple (depending on dimensionality).
        Width, height, and depth (if 3-D) of patches.

    max_number_of_patches:  integer or string
        Maximum number of patches returned.  If "all" is specified, then
        all patches in sequence (defined by the stride_length are extracted.

    stride_length:  integer or n-D tuple
        Defines the sequential patch overlap for max_number_of_patches = "all".
        Can be a image-dimensional vector or a scalar.

    mask_image:  ANTsImage (optional)
        Optional image specifying the sampling region for
        the patches when max_number_of_patches does not equal "all".
        The way we constrain patch selection using a mask is by forcing
        each returned patch to have a masked voxel at its center.

    random_seed: integer (optional)
        Seed value that allows reproducible patch extraction across runs.

    return_as_array: boolean
        Specifies the return type of the function.  If
        False (default) the return type is a list where each element is
        a single patch.  Otherwise the return type is an array of size
        dim( number_of_patches, patch_size ).

    randomize: boolean
        Boolean controlling whether we randomize indices when masking.

    Returns
    -------
    A list (or array) of patches.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> image_patches = extract_image_patches(image, patch_size=(32, 32))
    """

    if random_seed is not None:
        random.seed(random_seed)

    image_size = image.shape
    dimensionality = image.dimension

    if dimensionality != 2 and dimensionality != 3:
        raise ValueError("Unsupported dimensionality.")

    number_of_image_components = image.components

    if len(image_size) != len(patch_size):
        raise ValueError("Mismatch between the image size and the specified patch size.")

    if tuple(patch_size) > image_size:
        raise ValueError("Patch size is greater than the image size.")

    image_array = image.numpy()

    patch_list = []
    patch_array = np.empty([1, 1])
    mid_patch_index = tuple(np.int_(np.subtract(np.round(0.5 * (np.array(patch_size))), 1)))

    number_of_extracted_patches = max_number_of_patches

    if isinstance(max_number_of_patches, str) and max_number_of_patches.lower() == 'all':
        stride_length_tuple = stride_length
        if isinstance(stride_length, int):
            stride_length_tuple = tuple(np.multiply(np.ones_like(patch_size), stride_length))
        elif len(stride_length) != dimensionality:
            raise ValueError("stride_length is not a scalar or vector of length dimensionality.")
        elif np.any(np.less(stride_length, 1)):
            raise ValueError("stride_length elements must be positive integers.")

        number_of_extracted_patches = 1

        indices = []
        for d in range(dimensionality):
            indices.append(range(0, image_size[d] - patch_size[d] + 1, stride_length_tuple[d]))
            number_of_extracted_patches *= len(indices[d])

        if return_as_array:
            if number_of_image_components == 1:
                patch_array = np.zeros((number_of_extracted_patches, *patch_size))
            else:
                patch_array = np.zeros((number_of_extracted_patches, *patch_size, number_of_image_components))

        count = 0
        if dimensionality == 2:
            for i in indices[0]:
                for j in indices[1]:
                    start_index = (i, j)
                    end_index = tuple(np.add(start_index, patch_size))

                    if number_of_image_components == 1:
                        patch = image_array[start_index[0]:end_index[0],
                                            start_index[1]:end_index[1]]
                    else:
                        patch = image_array[start_index[0]:end_index[0],
                                            start_index[1]:end_index[1],:]

                    if return_as_array:
                        if number_of_image_components == 1:
                            patch_array[count, :, :] = patch
                        else:
                            patch_array[count, :, :, :] = patch
                    else:
                        patch_list.append(patch)

                    count += 1
        else:
            for i in indices[0]:
                for j in indices[1]:
                    for k in indices[2]:
                        start_index = (i, j, k)
                        end_index = tuple(np.add(start_index, patch_size))

                        if number_of_image_components == 1:
                            patch = image_array[start_index[0]:end_index[0],
                                                start_index[1]:end_index[1],
                                                start_index[2]:end_index[2]]
                        else:
                            patch = image_array[start_index[0]:end_index[0],
                                                start_index[1]:end_index[1],
                                                start_index[2]:end_index[2],:]

                        if return_as_array:
                            if number_of_image_components == 1:
                                patch_array[count, :, :, :] = patch
                            else:
                                patch_array[count, :, :, :, :] = patch
                        else:
                            patch_list.append(patch)

                        count += 1
    else:

        random_indices = np.zeros((max_number_of_patches, dimensionality), dtype=int)

        if mask_image != None:
            mask_array = mask_image.numpy()
            mask_array[np.where(mask_array != 0)] = 1

            # The way we constrain patch selection using a mask is by assuming that
            # each patch must have a masked voxel at its midPatchIndex.

            mask_indices = np.column_stack(np.where(mask_array != 0))
            for d in range(dimensionality):
                shifted_mask_indices = np.subtract(mask_indices[:, d], mid_patch_index[d])
                outside_indices = np.where(shifted_mask_indices < 0)
                mask_indices = np.delete(mask_indices, outside_indices, axis=0)
                shifted_mask_indices = np.add(mask_indices[:, d], mid_patch_index[d])
                outside_indices = np.where(shifted_mask_indices >= image_size[d] - 1)
                mask_indices = np.delete(mask_indices, outside_indices, axis=0)

            # After pruning the mask indices, which were originally defined in terms of the
            # mid_patch_index, we subtract the midPatchIndex so that it's now defined at the
            # corner for patch selection.

            for d in range(dimensionality):
                mask_indices[:, d] = np.subtract(mask_indices[:, d], mid_patch_index[d])

            number_of_extracted_patches = min(max_number_of_patches, mask_indices.shape[0])

            if randomize:
                random_indices = mask_indices[
                  random.sample(range(mask_indices.shape[0]), number_of_extracted_patches), :]
            else:
                random_indices = mask_indices

        else:

            for d in range(dimensionality):
                random_indices[:, d] = random.sample(range(image_size[d] - patch_size[d] + 1), max_number_of_patches)

        if return_as_array:
            if number_of_image_components == 1:
                patch_array = np.zeros((number_of_extracted_patches, *patch_size))
            else:
                patch_array = np.zeros((number_of_extracted_patches, *patch_size, number_of_image_components))

        start_index = np.zeros( dimensionality )
        for i in range(number_of_extracted_patches):
            start_index = random_indices[i, :]
            end_index = np.add(start_index, patch_size)

            if dimensionality == 2:
                if number_of_image_components == 1:
                    patch = image_array[start_index[0]:end_index[0],
                                        start_index[1]:end_index[1]]
                else:
                    patch = image_array[start_index[0]:end_index[0],
                                        start_index[1]:end_index[1], :]
            else:
                if number_of_image_components == 1:
                    patch = image_array[start_index[0]:end_index[0],
                                        start_index[1]:end_index[1],
                                        start_index[2]:end_index[2]]
                else:
                    patch = image_array[start_index[0]:end_index[0],
                                        start_index[1]:end_index[1],
                                        start_index[2]:end_index[2], :]

            if return_as_array:
                if dimensionality == 2:
                    if number_of_image_components == 1:
                        patch_array[i, :, :] = patch
                    else:
                        patch_array[i, :, :, :] = patch
                else:
                    if number_of_image_components == 1:
                        patch_array[i, :, :, :] = patch
                    else:
                        patch_array[i, :, :, :, :] = patch
            else:
                patch_list.append(patch)

    if return_as_array:
       return(patch_array)
    else:
       return(patch_list)
