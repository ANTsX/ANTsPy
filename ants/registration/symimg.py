

__all__ = ['symimg']

from tempfile import mktemp

from .reflect_image import reflect_image
from .interface import registration
from .apply_transforms import apply_transforms
from ..core import image_io as iio


def symimg(img):
    """
    Use registration and reflection to make an image symmetric

    ANTsR function: N/A

    Arguments
    ---------
    img : ANTsImage
        image to make symmetric

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.image_read( ants.get_ants_data('r16') , 'float')
    >>> simg = ants.symimg(img)
    """
    imgr = reflect_image(img, axis=0)
    imgavg = imgr * 0.5 + img
    for i in range(5):
        w1 = registration(imgavg, img, type_of_transform='SyN')
        w2 = registration(imgavg, imgr, type_of_transform='SyN')
        
        xavg = w1['warpedmovout']*0.5 + w2['warpedmovout']*0.5
        nada1 = apply_transforms(img, img, w1['fwdtransforms'], compose=w1['fwdtransforms'][0])
        nada2 = apply_transforms(img, img, w2['fwdtransforms'], compose=w2['fwdtransforms'][0])

        wavg = (iio.image_read(nada1) + iio.image_read(nada2)) * (-0.5)
        wavgfn = mktemp(suffix='.nii.gz')
        iio.image_write(wavg, wavgfn)
        xavg = apply_transforms(img, imgavg, wavgfn)

    return xavg


