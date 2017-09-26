

__all__ = ['mask_image']


def mask_image(image, mask, level=1, binarize=False):
    """
    Mask an input image by a mask image.  If the mask image has multiple labels,
    it is possible to specify which label(s) to mask at.
    
    ANTsR function: `maskImage`
    
    Arguments
    ---------
    image : ANTsImage
        Input image.

    mask : ANTsImage
        Mask or label image.

    level : scalar or tuple of scalars
        Level(s) at which to mask image. If vector or list of values, output image is non-zero at all locations where label image matches any of the levels specified.

    binarize : boolean
        whether binarize the output image
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> myimage = ants.image_read(ants.get_ants_data('r16'))
    >>> mask = ants.get_mask(myimage)
    >>> myimage_mask = ants.mask_image(myimage, mask, 3)
    >>> seg = ants.kmeans_segmentation(myimage, 3)
    >>> myimage_mask = ants.mask_image(myimage, seg['segmentation'], (1,3))
    """
    if not isinstance(level, (tuple,list)):
        image_out = image.clone()
        image_out[mask != level] = 0
        return image_out
    else:
        image_out = image.clone() * 0
        for mylevel in level:
            if binarize:
                image_out[mask == mylevel] = 1
            else:
                image_out[mask == mylevel] = image[mask == mylevel]
        return image_out

