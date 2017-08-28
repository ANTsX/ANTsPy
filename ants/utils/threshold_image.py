
 

__all__ = ['threshold_image']

from .process_args import _int_antsProcessArguments
from .. import lib


def threshold_image(img, low_thresh=None, high_thresh=None, inval=1, outval=0):
    """
    Threshold an ANTsImage
    """
    dim = img.dimension
    outimg = img.clone()
    args = [dim, img, outimg, low_thresh, high_thresh, inval, outval]
    processed_args = _int_antsProcessArguments(args)
    lib.ThresholdImage(processed_args)
    return outimg