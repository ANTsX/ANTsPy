
__all__ = ['weingarten_image_curvature']

from .. import core
from ..core import ants_image as iio
from .. import utils


def weingarten_image_curvature(image, sigma=1.0, opt='mean'):
    """
    Uses the weingarten map to estimate image mean or gaussian curvature

    ANTsR function: `weingartenImageCurvature`
    
    Arguments
    ---------
    image : ANTsImage
        image from which curvature is calculated
    
    sigma : scalar
        smoothing parameter
    
    opt : string
        mean by default, otherwise `gaussian` or `characterize`
    
    Returns
    -------
    ANTsImage

    Example
    -------
    >>> image = ants.image_read(ants.get_ants_data('mni'))
    >>> imagecurv = ants.weingarten_image_curvature(image)
    """
    if image.dimension not in {2,3}:
        raise ValueError('image must be 2D or 3D')

    if image.dimension == 2:
        d = image.shape
        temp = core.make_image(d, 10).numpy()
        for k in range(1,7):
            voxvals = image[:d[0],:d[1]]
            temp[:d[0],:d[1],k] = voxvals
        temp = core.from_numpy(temp)
        myspc = image.spacing
        myspc = list(myspc) + [min(myspc)]
        temp.set_spacing(myspc)
    else:
        temp = image.clone()

    optnum = 0
    if opt == 'gaussian': 
        optnum = 6
    if opt == 'characterize':
        optnum = 5

    libfn = utils.get_lib_fn('weingartenImageCurvature')
    mykout = libfn(temp.pointer, sigma, optnum)
    mykout = iio.ANTsImage(pixeltype=image.pixeltype, dimension=image.dimension,
                            components=image.components, pointer=mykout)
    if image.dimension == 3:
        return mykout
    elif image.dimension == 2:
        subarr = core.from_numpy(mykout.numpy()[:,:,5])
        return core.copy_image_info(image, subarr)
