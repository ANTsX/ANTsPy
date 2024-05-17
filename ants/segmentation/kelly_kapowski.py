"""
Kelly Kapowski algorithm with computing cortical thickness
"""

__all__ = ['kelly_kapowski']

import ants
from ants.internal import get_lib_fn, get_pointer_string, process_arguments


def kelly_kapowski(s, g, w, its=45, r=0.025, m=1.5, gm_label=2, wm_label=3, **kwargs):
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

    gm_label : integer
        label for gray matter in the segmentation image

    wm_label : integer
        label for white matter in the segmentation image

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
    if ants.is_image(s):
        s = s.clone('unsigned int')

    d = s.dimension
    outimg = g.clone() * 0.0
    kellargs = {'d': d,
                's': "[{},{},{}]".format(get_pointer_string(s),gm_label,wm_label),
                'g': g,
                'w': w,
                'c': "[{}]".format(its),
                'r': r,
                'm': m,
                'o': outimg}
    for k, v in kwargs.items():
        kellargs[k] = v

    processed_kellargs = process_arguments(kellargs)

    libfn = get_lib_fn('KellyKapowski')
    libfn(processed_kellargs)

    # Check thickness is not still all zeros
    if outimg.sum() == 0.0:
        raise RuntimeError("KellyKapowski failed to compute thickness")

    return outimg



