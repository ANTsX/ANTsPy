

__all__ = ['rgb_to_vector', 
           'vector_to_rgb', 
           'scalar_to_rgb']

import os
from tempfile import mktemp

import numpy as np

import ants
from ants.internal import get_lib_fn, process_arguments
from ants.decorators import image_method

def scalar_to_rgb(image, mask=None, filename=None, cmap='red', custom_colormap_file=None, 
                  min_input=None, max_input=None, min_rgb_output=None, max_rgb_output=None,
                  vtk_lookup_table=None):
    """
    Usage: ConvertScalarImageToRGB imageDimension inputImage outputImage mask colormap 
    [customColormapFile] [minimumInput] [maximumInput] [minimumRGBOutput=0] 
    [maximumRGBOutput=255] <vtkLookupTable>
    Possible colormaps: grey, red, green, blue, copper, jet, hsv, spring, summer, autumn, winter, hot, cool, overunder, custom
    
    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> img_color = ants.scalar_to_rgb(img, cmap='jet')
    """
    raise Exception('This function is currently not supported.')

@image_method
def rgb_to_vector(image):
    """
    Convert an RGB ANTsImage to a Vector ANTsImage

    Arguments
    ---------
    image : ANTsImage
        RGB image to be converted

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> mni = ants.image_read(ants.get_data('mni'))
    >>> mni_rgb = ants.scalar_to_rgb(mni)
    >>> mni_vector = mni.rgb_to_vector()
    >>> mni_rgb2 = mni.vector_to_rgb()
    """
    if image.pixeltype != 'unsigned char':
        image = image.clone('unsigned char')
    idim = image.dimension
    libfn = get_lib_fn('RgbToVector%i' % idim)
    new_ptr = libfn(image.pointer)
    new_img = ants.from_pointer(new_ptr)
    return new_img

@image_method
def vector_to_rgb(image):
    """
    Convert an Vector ANTsImage to a RGB ANTsImage

    Arguments
    ---------
    image : ANTsImage
        RGB image to be converted

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'), pixeltype='unsigned char')
    >>> img_rgb = ants.scalar_to_rgb(img.clone())
    >>> img_vec = img_rgb.rgb_to_vector()
    >>> img_rgb2 = img_vec.vector_to_rgb()
    """
    if image.pixeltype != 'unsigned char':
        image = image.clone('unsigned char')
    idim = image.dimension
    libfn = get_lib_fn('VectorToRgb%i' % idim)
    new_ptr = libfn(image.pointer)
    new_img = ants.from_pointer(new_ptr)
    return new_img

