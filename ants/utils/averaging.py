import os
from tempfile import mktemp

import numpy as np

from .. import utils
from ..core import ants_image as iio
from ..core import ants_image_io as iio2
from .. import registration as reg

########################
def average_images( x, normalize=True, mask=None, imagetype=0 ):
    """
    average a list of images

    images will be resampled automatically to the largest image space;
    this is not a registration so images should be in the same physical
    space to begin with.

    x : a list containing either filenames or antsImages 

    normalize : boolean

    mask : None or Otsu (string); this will perform a masked averaging which can 
        be useful when images have only partial coverage

    imagetype : integer
        choose 0/1/2/3 mapping to scalar/vector/tensor/time-series

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> x0=[ ants.get_data('r16'), ants.get_data('r27'), ants.get_data('r62'), ants.get_data('r64') ]
    >>> x1=[]
    >>> for k in range(len(x0)):
    >>>     x1.append( ants.image_read( x0[k] ) )
    >>> avg=ants.average_images(x0)
    >>> avg1=ants.average_images(x1)
    >>> avg2=ants.average_images(x1,mask=True)
	>>> avg3=ants.average_images(x1,mask=True,normalize=True)
    """
    import numpy as np

    def gli( y, normalize=False ):
        if isinstance(y,str):
            y=iio2.image_read(y)
        if normalize:
            y=y/y.mean()
        return y

    biggest=0
    biggestind=0
    for k in range( len( x ) ):
        locimg = gli( x[k], False )
        sz=np.prod( locimg.shape )
        if sz > biggest:
            biggest=sz
            biggestind=k

    avg = gli( x[biggestind], False ) * 0
    scl = float( 1.0 / len(x))
    if mask is not None:
        sumimg = gli( x[biggestind], False ) * 0

    for k in range( len( x ) ):
        locimg = gli( x[k], normalize )
        temp = reg.resample_image_to_target( locimg, avg, interp_type='linear', imagetype=imagetype )
        avg = avg + temp
        if mask is not None:
            sumimg = sumimg + utils.threshold_image(temp,'Otsu',1)

    if mask is None:
        avg = avg * scl
    else:
        nonzero = sumimg > 0
        avg[nonzero] = avg[nonzero] / sumimg[nonzero]
    return avg        

