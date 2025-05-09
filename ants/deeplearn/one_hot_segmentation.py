__all__ = ["segmentation_to_one_hot,"
           "one_hot_to_segmentation"]

import ants
import numpy as np

def segmentation_to_one_hot(segmentations_array,
                            segmentation_labels=None,
                            channel_first_ordering=False):

    """
    Basic one-hot transformation of segmentations array

    Arguments
    ---------
    segmentations_array : numpy array
        multi-label numpy array

    segmentation_labels : tuple or list
        Note that a background label (typically 0) needs to be included.
        
    channel_first_ordering : bool
        Specifies the ordering of the dimensions.

    Returns
    -------
    A numpy array where the segmentation labels are expanded in a one-hot
    fashion in the channels dimension.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> seg = ants.kmeans_segmentation(image, 3)['segmentation']
    >>> one_hot = segmentation_to_one_hot(seg.numpy().astype('int'))
    """

    if segmentation_labels is None:
        segmentation_labels = np.unique(segmentations_array)

    number_of_labels = len(segmentation_labels)
    if number_of_labels < 2:
        raise ValueError("At least two segmentation labels need to be specified.")

    image_dimension = len(segmentations_array.shape)

    one_hot_array = np.zeros((*segmentations_array.shape, number_of_labels))
    for i in range(number_of_labels):
        per_label = np.zeros_like(segmentations_array)
        per_label[segmentations_array == segmentation_labels[i]] = 1
        if image_dimension == 2:
            one_hot_array[:,:,i] = per_label
        elif image_dimension == 3:
            one_hot_array[:,:,:,i] = per_label
        else:
            raise ValueError("Unrecognized image dimensionality.")

    if channel_first_ordering:  
        if image_dimension == 2:
            one_hot_array = one_hot_array.transpose((2, 0, 1))
        elif image_dimension == 3:
            one_hot_array = one_hot_array.transpose((3, 0, 1, 2))

    return one_hot_array


def one_hot_to_segmentation(one_hot_array,
                            domain_image,
                            channel_first_ordering=False):

    """
    Inverse of basic one-hot transformation of segmentations array

    Arguments
    ---------
    one_hot_array : an array
        Numpy array where the channel dimension contains the one-hot
        encoding.

    domain_image : ANTs image
        Defines the geometry of the returned probability images

    channel_first_ordering : bool
        Specifies the ordering of the dimensions.
        
    Returns
    -------
    List of probability images.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    """

    number_of_labels = one_hot_array.shape[-1]
    if channel_first_ordering:
        number_of_labels = one_hot_array.shape[0]

    image_dimension = domain_image.dimension

    probability_images = list()
    for label in range(number_of_labels):
        if image_dimension == 2:
            if channel_first_ordering:
                image_array = np.squeeze(one_hot_array[label,:,:])
            else:         
                image_array = np.squeeze(one_hot_array[:,:,label])
        else:
            if channel_first_ordering:
                image_array = np.squeeze(one_hot_array[label,:,:,:])
            else:         
                image_array = np.squeeze(one_hot_array[:,:,:,label])
        probability_images.append(
            ants.from_numpy_like(image_array, domain_image))

    return probability_images

