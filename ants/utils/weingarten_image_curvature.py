
__all__ = ['weingarten_image_curvature']

from .. import core
from ..core import ants_image as iio
from .. import lib


def weingarten_image_curvature(img, sigma=1.0, opt='mean'):
    if img.dimension not in {2,3}:
        raise ValueError('image must be 2D or 3D')

    if img.dimension == 2:
        d = img.shape
        temp = core.make_image(d, 10).numpy()
        for k in range(1,7):
            voxvals = img[:d[0],:d[1]]
            temp[:d[0],:d[1],k] = voxvals
        temp = core.from_numpy(temp)
        myspc = img.spacing
        myspc = list(myspc) + [min(myspc)]
        temp.set_spacing(myspc)
    else:
        temp = img.clone()

    optnum = 0
    if opt == 'gaussian': 
        optnum = 6
    if opt == 'characterize':
        optnum = 5

    mykout = lib.weingartenImageCurvature(temp._img, sigma, optnum)
    mykout = iio.ANTsImage(mykout)
    if img.dimension == 3:
        return mykout
    elif img.dimension == 2:
        subarr = core.from_numpy(mykout.numpy()[:,:,5])
        return core.copy_image_info(img, subarr)