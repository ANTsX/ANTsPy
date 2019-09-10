
__all__ = ['create_tiled_mosaic']

import os
from tempfile import mktemp

from PIL import Image

from .. import utils
from ..core import ants_image_io as iio2


def create_tiled_mosaic(image, rgb=None, mask=None, overlay=None,
                        output=None, alpha=1., direction=0,
                        pad_or_crop=None, slices=None,
                        flip_slice=None, permute_axes=False):
    """
    Create a tiled mosaic of 2D slice images from a 3D ANTsImage.

    ANTsR function : N/A
    ANTs function  : `createTiledMosaic`

    Arguments
    ---------
    image : ANTsImage
        base image to visualize

    rgb : ANTsImage
        optional overlay image to display on top of base image

    mask : ANTsImage
        optional mask image

    alpha : float
        alpha value for rgb/overlay image

    direction : integer or string
        which axis to visualize
        options: 0, 1, 2, 'x', 'y', 'z'

    pad_or_crop : list of 2-tuples
        padding or cropping values for each dimension and each side.
        - to crop the X dimension, use the following:
            pad_or_crop = [(10,10), 0, 0]
        - to pad the X dimension, use the following:
            pad_or_crop = [(-10,-10), 0, 0]

    slices : list/numpy.ndarray or integer or 3-tuple
        if list or numpy.ndarray:
            slices to use
        if integer:
            number of slices to incremenet
        if 3-tuple:
            (# slices to increment, min slice, max slice)

    flip_slice : 2-tuple of boolean
        (whether to flip X direction, whether to flip Y direction)

    permute_axes : boolean
        whether to permute axes

    output : string
        output filename where mosaic image will be saved.
        If not given, this function will save to a temp file,
        then return the image as a PIL.Image object

    ANTs
    ----
     -i, --input-image inputImageFilename
     -r, --rgb-image rgbImageFilename
     -x, --mask-image maskImageFilename
     -a, --alpha value
     -e, --functional-overlay [rgbImageFileName,maskImageFileName,<alpha=1>]
     -o, --output tiledMosaicImage
     -t, --tile-geometry RxC
     -d, --direction 0/1/2/x/y/(z)
     -p, --pad-or-crop padVoxelWidth
                       [padVoxelWidth,<constantValue=0>]
                       [lowerPadding[0]xlowerPadding[1],upperPadding[0]xupperPadding[1],constantValue]
     -s, --slices Slice1xSlice2xSlice3...
                  numberOfSlicesToIncrement
                  [numberOfSlicesToIncrement,<minSlice=0>,<maxSlice=lastSlice>]
     -f, --flip-slice flipXxflipY
     -g, --permute-axes doPermute
     -h

    CreateTiledMosaic -i OAS1_0457_MR1_mpr_n3_anon_sbj_111BrainSegmentation0N4 . nii . gz \
    -r OAS1_0457_MR1_mpr_n3_anon_sbj_111CorticalThickness_hot . nii . gz \
    -x OAS1_0457_MR1_mpr_n3_anon_sbj_111CorticalThickness_mask . nii . gz \
    -o OAS1_0457_MR1_mpr_n3_anon_sbj_111_tiledMosaic . png \
    -a 1.0 -t -1 x8 -d 2 -p [ -15x -50 , -15x -30 ,0] -s [2 ,100 ,160]

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('ch2'))
    >>> plt = ants.create_tiled_mosaic(image)
    """
    # image needs to be unsigned char
    if image.pixeltype != 'unsigned char':
        # transform between 0 and 255.
        image = (image - image.max()) / (image.max() - image.min())
        image = image * 255.
        image = image.clone('unsigned char')

    output_is_temp = False
    if output is None:
        output_is_temp = True
        output = mktemp(suffix='.jpg')

    if rgb is None:
        rgb = image.clone()


    imagepath = mktemp(suffix='.nii.gz')
    iio2.image_write(image, imagepath)
    rgbpath = mktemp(suffix='.nii.gz')
    iio2.image_write(rgb, rgbpath)
    args = {
        'i': imagepath,
        'r': rgbpath,
        'o': output,
        'x': mask,
        'e': overlay,
        'a': alpha,
        'd': direction
    }

    processed_args = utils._int_antsProcessArguments(args)

    libfn = utils.get_lib_fn('CreateTiledMosaic')

    libfn(processed_args)

    outimage = Image.open(output)
    if output_is_temp:
        os.remove(output)
    return outimage
