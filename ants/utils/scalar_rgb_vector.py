

__all__ = ['rgb_to_vector', 'vector_to_rgb', 'scalar_to_rgb']

import os
from tempfile import mktemp

import numpy as np

from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2


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
    >>> img_color = img.scalar_to_rgb(cmap='jet')
    """

    if filename is None:
        file_is_temp = True
        filename = mktemp(suffix='.png')
    else:
        file_is_temp = False

    args = 'imageDimension inputImage outputImage mask colormap'.split(' ')
    args[0] = image.dimension

    if isinstance(image, iio.ANTsImage):
        tmpimgfile = mktemp(suffix='.nii.gz')
        image.to_file(tmpimgfile)
    elif isinstance(image, str):
        tmpimgfile = image
    args[1] = tmpimgfile
    args[2] = filename
    args[3] = mask if mask is not None else image.new_image_like(np.ones(image.shape))
    args[4] = cmap
    if custom_colormap_file is not None:
        args.append('customColormapFile=%s' % custom_colormap_file)
    if min_input is not None:
        args.append('minimumInput=%f' % min_input)
    if max_input is not None:
        args.append('maximumInput=%f' % max_input)
    if min_rgb_output is not None:
        args.append('minRGBOutput=%f' % min_rgb_output)
    if max_rgb_output is not None:
        args.append('maxRGBOutput=%f' % min_rgb_output)
    if vtk_lookup_table is not None:
        vtk_lookup_table = mktemp(suffix='.csv')
        args.append('vtkLookupTable=%s' % vtk_lookup_table)
    
    processed_args = utils._int_antsProcessArguments(args)
    libfn = utils.get_lib_fn('ConvertScalarImageToRGB')
    libfn(processed_args)

    if file_is_temp:
        outimg = iio2.image_read(filename, pixeltype=None)
        # clean up temp files
        os.remove(filename)
        os.remove(tmpimgfile)

        return outimg
    else:
        os.remove(tmpimgfile)


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
    >>> mni_rgb = mni.scalar_to_rgb()
    >>> mni_vector = mni.rgb_to_vector()
    >>> mni_rgb2 = mni.vector_to_rgb()
    """
    if image.pixeltype != 'unsigned char':
        image = image.clone('unsigned char')
    idim = image.dimension
    libfn = utils.get_lib_fn('RgbToVector%i' % idim)
    new_ptr = libfn(image.pointer)
    new_img = iio.ANTsImage(pixeltype=image.pixeltype, dimension=image.dimension, 
                            components=3, pointer=new_ptr, is_rgb=False)
    return new_img


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
    >>> img_rgb = img.clone().scalar_to_rgb()
    >>> img_vec = img_rgb.rgb_to_vector()
    >>> img_rgb2 = img_vec.vector_to_rgb()
    """
    if image.pixeltype != 'unsigned char':
        image = image.clone('unsigned char')
    idim = image.dimension
    libfn = utils.get_lib_fn('VectorToRgb%i' % idim)
    new_ptr = libfn(image.pointer)
    new_img = iio.ANTsImage(pixeltype=image.pixeltype, dimension=image.dimension, 
                            components=3, pointer=new_ptr, is_rgb=True)
    return new_img

