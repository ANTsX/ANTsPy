
 

__all__ = ['kmeans_segmentation']

from .atropos import atropos
from .. import utils

def kmeans_segmentation(img, k, kmask=None, mrf=0.1):
    """
    K-means image segmentation that is a wrapper around `ants.atropos`

    ANTsR function: `kmeansSegmentation`
    
    Arguments
    ---------
    img : ANTsImage 
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
    >>> fi = ants.image_read(ants.get_ants_data('r16'), 'float')
    >>> fi = ants.n3_bias_field_correction(fi, 2)
    >>> seg = ants.kmeans_segmentation(fi, 3)
    """
    dim = img.dimension
    kmimg = utils.iMath(img, 'Normalize')
    if kmask is None:
        kmask = utils.get_mask(kmimg, 0.01, 1, cleanup=2)
    kmask = utils.iMath(kmask, 'FillHoles')
    nhood = 'x'.join(['1']*dim)
    mrf = '[%s,%s]' % (str(mrf), nhood)
    kmimg = atropos(a = kmimg, m = mrf, c = '[5,0]', i = 'kmeans[%s]'%(str(k)), x = kmask)
    kmimg['segmentation'] = kmimg['segmentation'].clone(img.pixeltype)
    return kmimg


