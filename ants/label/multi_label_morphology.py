"""
Morphology operations on multi-label ANTsImage types
"""

__all__ = ['multi_label_morphology']

import numpy as np

def multi_label_morphology(image, operation, radius, dilation_mask=None, label_list=None, force=False):
    """
    Morphology on multi label images.
    
    Wraps calls to iMath binary morphology. Additionally, dilation and closing operations preserve
    pre-existing labels. The choices of operation are:
    
    Dilation: dilates all labels sequentially, but does not overwrite original labels.
    This reduces dependence on the intensity ordering of adjoining labels. Ordering dependence
    can still arise if two or more labels dilate into the same space - in this case, the label
    with the lowest intensity is retained. With a mask, dilated labels are multiplied by the
    mask and then added to the original label, thus restricting dilation to the mask region.
    
    Erosion: Erodes labels independently, equivalent to calling iMath iteratively.
    
    Closing: Close holes in each label sequentially, but does not overwrite original labels.
    
    Opening: Opens each label independently, equivalent to calling iMath iteratively.
    
    Arguments
    ---------
    image : ANTsImage
        Input image should contain only 0 for background and positive integers for labels.
    
    operation : string
        One of MD, ME, MC, MO, passed to iMath.
    
    radius : integer
        radius of the morphological operation.
    
    dilation_mask : ANTsImage
        Optional binary mask to constrain dilation only (eg dilate cortical label into WM).
    
    label_list : list or tuple or numpy.ndarray
        Optional list of labels, to perform operation upon. Defaults to all unique 
        intensities in image.

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> labels = ants.get_mask(img,1,150) + ants.get_mask(img,151,225) * 2
    >>> labels_dilated = ants.multi_label_morphology(labels, 'MD', 2)
    >>> # should see original label regions preserved in dilated version
    >>> # label N should have mean N and 0 variance
    >>> print(ants.label_stats(labels_dilated, labels))
    """
    if (label_list is None) or (len(label_list) == 1):
        label_list = np.sort(np.unique(image[image > 0]))
        if (len(label_list) > 200) and (not force):
            raise ValueError('More than 200 labels... Make sure the image is discrete'
                             ' and call this function again with `force=True` if you'
                             ' really want to do this.')

    image_binary = image.clone()
    image_binary[image_binary > 1] = 1

    # Erosion / opening is simply a case of looping over the input labels
    if (operation == 'ME') or (operation == 'MO'):
        output = image.clone()

        for current_label in label_list:
            output = output.iMath(operation, radius, current_label)

        return output

    if dilation_mask is not None:
        if int(dilation_mask.max()) != 1:
            raise ValueError('Mask is either empty or not binary')

    output = image.clone()
    for current_label in label_list:
        current_label_region = image.threshold_image(current_label, current_label)
        other_labels = output - current_label_region
        clab_binary_morphed = current_label_region.iMath(operation, radius, 1)

        if (operation == 'MD') and (dilation_mask is not None):
            clab_binary_morphed_nooverlap = current_label_region + dilation_mask * clab_binary_morphed - other_labels
            clab_binary_morphed_nooverlap = clab_binary_morphed_nooverlap.threshold_image(1, 2)
        else:
            clab_binary_morphed_nooverlap = clab_binary_morphed - other_labels
            clab_binary_morphed_nooverlap = clab_binary_morphed_nooverlap.threshold_image(1, 1)

        output = output + clab_binary_morphed_nooverlap * current_label

    return output

