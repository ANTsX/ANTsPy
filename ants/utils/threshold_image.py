
 

__all__ = ['threshold_image']

from .process_args import _int_antsProcessArguments
from .. import utils


def threshold_image(image, low_thresh=None, high_thresh=None, inval=1, outval=0):
    """
    Converts a scalar image into a binary image by thresholding operations

    ANTsR function: `thresholdImage`

    Arguments
    ---------
    image : ANTsImage
        Input image to operate on
    
    low_thresh : scalar (optional)
        Lower edge of threshold window
    
    hight_thresh : scalar (optional)
        Higher edge of threshold window
    
    inval : scalar
        Output value for image voxels in between lothresh and hithresh
    
    outval : scalar
        Output value for image voxels lower than lothresh or higher than hithresh
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> timage = ants.threshold_image(image, 0.5, 1e15)
    """
    dim = image.dimension
    outimage = image.clone()
    args = [dim, image, outimage, low_thresh, high_thresh, inval, outval]
    processed_args = _int_antsProcessArguments(args)
    libfn = utils.get_lib_fn('ThresholdImage')
    libfn(processed_args)
    return outimage