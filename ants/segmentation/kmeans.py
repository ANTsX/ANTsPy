


__all__ = ['kmeans_segmentation']

from .atropos import atropos
from .. import utils

def kmeans_segmentation(image, k, kmask=None, mrf=0.1):
    """
    K-means image segmentation that is a wrapper around `ants.atropos`

    ANTsR function: `kmeansSegmentation`

    Arguments
    ---------
    image : ANTsImage
        input image

    k : integer
        integer number of classes

    kmask : ANTsImage (optional)
        segment inside this mask

    mrf : scalar
        smoothness, higher is smoother

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> fi = ants.n3_bias_field_correction(fi, 2)
    >>> seg = ants.kmeans_segmentation(fi, 3)
    """
    dim = image.dimension
    kmimage = utils.iMath(image, 'Normalize')
    if kmask is None:
        kmask = utils.get_mask(kmimage, 0.01, 1, cleanup=2)
    kmask = utils.iMath(kmask, 'FillHoles').threshold_image(1,2)
    nhood = 'x'.join(['1']*dim)
    mrf = '[%s,%s]' % (str(mrf), nhood)
    kmimage = atropos(a = kmimage, m = mrf, c = '[5,0]', i = 'kmeans[%s]'%(str(k)), x = kmask)
    kmimage['segmentation'] = kmimage['segmentation'].clone(image.pixeltype)
    return kmimage
