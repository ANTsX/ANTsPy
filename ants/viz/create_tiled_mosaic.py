
__all__ = ['create_tiled_mosaic']

import os
from tempfile import mktemp 

from PIL import Image

from .. import lib
from .. import utils

def create_tiled_mosaic(img, output=None, rgb=None, mask=None, overlay=None,
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
    """
    output_is_temp = False
    if output is None:
        output_is_temp = True
        output = mktemp(suffix='.png')

    img = img.clone('float')

    args = {
        'i': img,
        'o': output
    }

    processed_args = utils._int_antsProcessArguments(args)
    retval = lib.CreateTiledMosaic(processed_args)

    if retval != 0:
        raise ValueError('Non-zero exit status')

    outimg = Image.open(output)
    if output_is_temp:
        os.remove(output)
    return outimg
