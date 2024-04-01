
__all__ = ['prior_based_segmentation']

from ..core import ants_image as iio
from .atropos import atropos


def prior_based_segmentation(image, priors, mask, priorweight=0.25, mrf=0.1, iterations=25):
    """
    Spatial prior-based image segmentation.
    
    Markov random field regularized, prior-based image segmentation that is a 
    wrapper around atropos (see ANTs and related publications).

    ANTsR function: `priorBasedSegmentation`

    Arguments
    ---------
    image : ANTsImage or list/tuple of ANTsImage types
        input image or image list for multivariate segmentation

    priors : list/tuple of ANTsImage types
        list of priors that cover the number of classes

    mask : ANTsImage
        segment inside this mask

    prior_weight : scalar
        usually 0 (priors used for initialization only), 0.25 or 0.5.

    mrf : scalar
        regularization, higher is smoother, a numerical value in range 0.0 to 0.2

    iterations : integer
        maximum number of iterations.  could be a large value eg 25.

    Returns
    -------
    dictionary with the following key/value pairs:
        `segmentation`: ANTsImage 
            actually segmented image
        
        `probabilityimages` : list of ANTsImage types
            one image for each segmentation class

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16'))
    >>> seg = ants.kmeans_segmentation(fi,3)
    >>> mask = ants.threshold_image(seg['segmentation'], 1, 1e15)
    >>> priorseg = ants.prior_based_segmentation(fi, seg['probabilityimages'], mask, 0.25, 0.1, 3)
    """
    if isinstance(image, iio.ANTsImage):
        dim = image.dimension
    elif isinstance(image, (tuple,list)) and (isinstance(image[0], iio.ANTsImage)):
        dim = image[0].dimension
    else:
        raise ValueError('image argument must be ANTsImage or list/tuple of ANTsImage types')

    nhood = 'x'.join(['1']*dim)
    mrf = '[%s,%s]' % (str(mrf), nhood)
    conv = '[%s,0]' % (str(iterations))

    pseg = atropos(a=image, m=mrf, c=conv, i=priors, x=mask, priorweight=priorweight)

    return pseg


