
 

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
    Crop an image using a label ANTsImage.

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16') )
    >>> cropped = ants.crop_image(fi)
    >>> cropped = ants.crop_image(fi, fi, 100 )

    Dev Note
    --------
    - Validated w/ ANTsR that it returns the same result,
      but plotting the resulting image in ANTsPy looks like
      gibberish. Need to understanding the plotting tricks used.
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

