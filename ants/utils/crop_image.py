
 

__all__ = ['crop_image', 
           'crop_indices',
           'decrop_image']


from .get_mask import get_mask
from ..core import ants_image as iio
from .. import lib

_crop_image_dict = {
    2: lib.cropImageF2,
    3: lib.cropImageF3
}


def crop_image(image, label_image=None, label=1):
    """
    Use a label image to crop a smaller ANTsImage from within a larger ANTsImage

    ANTsR function: `cropImage`
    
    Arguments
    ---------
    image : ANTsImage  
        image to crop
    
    label_image : ANTsImage
        imge with label values. If not supplied, estimated from data.
    
    label : integer   
        the label value to use

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16') )
    >>> cropped = ants.crop_image(fi)
    >>> cropped = ants.crop_image(fi, fi, 100 )
    """
    inpixeltype = image.pixeltype
    if image.pixeltype != 'float':
        image = image.clone('float')

    if label_image is None:
        label_image = get_mask(image)

    if label_image.pixeltype != 'float':
        label_image = label_image.clone('float')

    crop_image_fn = _crop_image_dict[image.dimension]
    itkimage = crop_image_fn(image._img, label_image._img, label, 0, [], [])
    return iio.ANTsImage(itkimage).clone(inpixeltype)


def crop_indices(image, lowerind, upperind):
    """
    Create a proper ANTsImage sub-image by indexing the image with indices. 
    This is similar to but different from array sub-setting in that 
    the resulting sub-image can be decropped back into its place without 
    having to store its original index locations explicitly.
    
    ANTsR function: `cropIndices`

    Arguments
    ---------
    image : ANTsImage  
        image to crop
    
    lowerind : list/tuple of integers  
        vector of lower index, should be length image dimensionality
    
    upperind : list/tuple of integers
        vector of upper index, should be length image dimensionality
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> fi = ants.image_read( ants.get_ants_data("r16"))
    >>> cropped = ants.crop_indices( fi, (10,10), (100,100) )
    >>> cropped = ants.smooth_image( cropped, 5 )
    >>> decropped = ants.decrop_image( cropped, fi )
    """
    if image.pixeltype != 'float':
        inpixeltype = image.pixeltype
        image = image.clone('float')

    if (image.dimension != len(lowerind)) or (image.dimension != len(upperind)):
        raise ValueError('image dimensionality and index length must match')

    crop_image_fn = _crop_image_dict[image.dimension]
    itkimage = crop_image_fn(image._img, image._img, 1, 2, lowerind, upperind)
    return iio.ANTsImage(itkimage).clone(inpixeltype)


def decrop_image(cropped_image, full_image):
    """
    The inverse function for `ants.crop_image`

    ANTsR function: `decropImage`
    
    Arguments
    ---------
    cropped_image : ANTsImage
        cropped image

    full_image : ANTsImage
        image in which the cropped image will be put back

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> fi = ants.image_read(ants.get_ants_data('r16'))
    >>> mask = ants.get_mask(fi)
    >>> cropped = ants.crop_image(fi, mask, 1)
    >>> cropped = ants.smooth_image(cropped, 1)
    >>> decropped = ants.decrop_image(cropped, fi)
    """
    if cropped_image.pixeltype != 'float':
        inpixeltype= cropped_image.pixeltype
        cropped_image = cropped_image.clone('float')
    if full_image.pixeltype != 'float':
        full_image = full_image.clone('float')

    crop_image_fn = _crop_image_dict[cropped_image.dimension]
    itkimage = crop_image_fn(cropped_image._img, full_image._img, 1, 1, [], [])
    return iio.ANTsImage(itkimage).clone(inpixeltype)

