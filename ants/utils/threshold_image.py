
 

__all__ = ['threshold_image']

from .process_args import _int_antsProcessArguments
from .. import utils


def threshold_image(image, low_thresh=None, high_thresh=None, inval=1, outval=0, binary=True):
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

    binary : boolean
        if true, returns binary thresholded image
        if false, return binary thresholded image multiplied by original image
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> timage = ants.threshold_image(image, 0.5, 1e15)
    """
    if high_thresh is None:
        high_thresh = image.max() + 0.01
    if low_thresh is None:
        low_thresh = image.min() - 0.01
    dim = image.dimension
    outimage = image.clone()
    args = [dim, image, outimage, low_thresh, high_thresh, inval, outval]
    processed_args = _int_antsProcessArguments(args)
    libfn = utils.get_lib_fn('ThresholdImage')
    libfn(processed_args)
    if binary:
        return outimage
    else:
        return outimage*image

