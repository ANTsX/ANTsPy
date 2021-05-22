"""
Kelly Kapowski algorithm with computing cortical thickness
"""

__all__ = ['kelly_kapowski']

from ..core import ants_image as iio
from .. import utils


def kelly_kapowski(s, g, w, its=45, r=0.025, m=1.5, **kwargs):
    """
    Compute cortical thickness using the DiReCT algorithm.

    Diffeomorphic registration-based cortical thickness based on probabilistic
    segmentation of an image.  This is an optimization algorithm.


    Arguments
    ---------
    s : ANTsimage
        segmentation image

    g : ANTsImage
        gray matter probability image

    w : ANTsImage
        white matter probability image

    its : integer
        convergence params - controls iterations

    r : scalar
        gradient descent update parameter

    m : scalar
        gradient field smoothing parameter

    kwargs : keyword arguments
        anything else, see KellyKapowski help in ANTs

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.image_read( ants.get_ants_data('r16') ,2)
    >>> img = ants.resample_image(img, (64,64),1,0)
    >>> mask = ants.get_mask( img )
    >>> segs = ants.kmeans_segmentation( img, k=3, kmask = mask)
    >>> thick = ants.kelly_kapowski(s=segs['segmentation'], g=segs['probabilityimages'][1],
                                    w=segs['probabilityimages'][2], its=45,
                                    r=0.5, m=1)
    """
    if isinstance(s, iio.ANTsImage):
        s = s.clone('unsigned int')

    d = s.dimension
    outimg = g.clone()
    kellargs = {'d': d,
                's': s,
                'g': g,
                'w': w,
                'c': its,
                'r': r,
                'm': m,
                'o': outimg}
    for k, v in kwargs.items():
        kellargs[k] = v

    processed_kellargs = utils._int_antsProcessArguments(kellargs)

    libfn = utils.get_lib_fn('KellyKapowski')
    libfn(processed_kellargs)
    return outimg



