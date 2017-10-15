
__all__ = ['create_tiled_mosaic']

import os
from tempfile import mktemp 

from PIL import Image

from .. import utils
from ..core import ants_image_io as iio2

def create_tiled_mosaic(img, rgb=None, output=None,  mask=None, overlay=None,
                        alpha=1., direction=0, 
                        pad_or_crop=None, slices=None,
                        flip_slice=None, permute_axes=False):
    """
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

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('ch2'))
    >>> plt = ants.create_tiled_mosaic(img)
    """
    # img needs to be unsigned char
    if img.pixeltype != 'unsigned char':
        # transform between 0 and 255.
        img = (img - img.max()) / (img.max() - img.min())
        img = img * 255.
        img = img.clone('unsigned char')
    
    output_is_temp = False
    if output is None:
        output_is_temp = True
        output = mktemp(suffix='.jpg')

    if rgb is None:
        rgb = img.clone()


    imgpath = mktemp(suffix='.nii.gz')
    iio2.image_write(img, imgpath)
    rgbpath = mktemp(suffix='.nii.gz')
    iio2.image_write(rgb, rgbpath)
    args = {
        'i': imgpath,
        'r': rgbpath,
        'o': output
    }

    processed_args = utils._int_antsProcessArguments(args)
    libfn = utils.get_lib_fn('CreateTiledMosaic')
    
    libfn(processed_args)

    outimg = Image.open(output)
    if output_is_temp:
        os.remove(output)
    return outimg
