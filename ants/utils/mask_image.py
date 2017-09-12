

__all__ = ['mask_image']


def mask_image(img, mask, level=1, binarize=False):
    """
    Mask an input image by a mask image.  If the mask image has multiple labels,
    it is possible to specify which label(s) to mask at.
    
    ANTsR function: `maskImage`
    
    Arguments
    ---------
    img : ANTsImage
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
    >>> myimg = ants.image_read(ants.get_ants_data('r16'))
    >>> mask = ants.get_mask(myimg)
    >>> myimg_mask = ants.mask_image(myimg, mask, 3)
    >>> seg = ants.kmeans_segmentation(myimg, 3)
    >>> myimg_mask = maskImage(myimg, seg['segmentation'], (1,3))
    """
    if not isinstance(level, (tuple,list)):
        img_out = img.clone()
        img_out[mask != level] = 0
        return img_out
    else:
        img_out = img.clone() * 0
        for mylevel in level:
            if binarize:
                img_out[mask == mylevel] = 1
            else:
                img_out[mask == mylevel] = img[mask == mylevel]
        return img_out

