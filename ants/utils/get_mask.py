
 

__all__ = ['get_mask']

from ..core import ants_image as iio
from .threshold_image import threshold_image
from .label_clusters import label_clusters
from .iMath import iMath
from .. import lib


def get_mask(img, low_thresh=None, high_thresh=None, cleanup=2):
    """
    Calculate a binary mask from an ANTsImage
    """
    cleanup = int(cleanup)
    if isinstance(img, iio.ANTsImage):
        if img.pixeltype != 'float':
            img = img.clone('float')

    if low_thresh is None:
        low_thresh = img.mean()
    if high_thresh is None:
        high_thresh = img.max()

    mask_img = threshold_image(img, low_thresh, high_thresh)
    if cleanup > 0:
        mask_img = iMath(mask_img, 'ME', cleanup)
        mask_img = iMath(mask_img, 'GetLargestComponent')
        mask_img = iMath(mask_img, 'MD', cleanup)
        mask_img = iMath(mask_img, 'FillHoles')
        while ((mask_img.min() == mask_img.max()) and (cleanup > 0)):
            cleanup = cleanup - 1
            mask_img = threshold_image(img, low_thresh, high_thresh)
            if cleanup > 0:
                mask_img = iMath(mask_img, 'ME', cleanup)
                mask_img = iMath(mask_img, 'MD', cleanup)
                mask_img = iMath(mask_img, 'FillHoles')

            #if cleanup == 0:
            #    clustlab = label_clusters(mask_img, 1)
            #    mask_img = threshold_image(clustlab, 1, 1)

    return mask_img







