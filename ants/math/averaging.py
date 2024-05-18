import os
from tempfile import mktemp

import numpy as np

import ants

__all__ = ['average_images']


def average_images( x, normalize=True, mask=None, imagetype=0, sum_image_threshold=3, return_sum_image=False, verbose=False ):
    """
    average a list of images

    images will be resampled automatically to the largest image space;
    this is not a registration so images should be in the same physical
    space to begin with.

    x : a list containing either filenames or antsImages 

    normalize : boolean

    mask : None or integer; this will perform a masked averaging which can 
        be useful when images have only partial coverage. integer greater 
        than zero will perform morphological closing.

    imagetype : integer
        choose 0/1/2/3 mapping to scalar/vector/tensor/time-series

    sum_image_threshold : integer
        only average regions with overlap greater than or equal to this value

    return_sum_image : boolean
        returns the average and the image that show ROI overlap; primarily for debugging

    verbose : boolean
        will print progress

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
    >>> avg2=ants.average_images(x1,mask=0)
	>>> avg3=ants.average_images(x1,mask=1,normalize=True)
    """
    import numpy as np

    def gli( y, normalize=False ):
        if isinstance(y,str):
            y=ants.image_read(y)
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
        if verbose and k % 20 == 0:
            print( str(k)+'...', end='',flush=True)
        locimg = gli( x[k], normalize )
        temp = ants.resample_image_to_target( locimg, avg, interp_type='linear', imagetype=imagetype )
        avg = avg + temp
        if mask is not None:
            fgmask = ants.threshold_image(temp,'Otsu',1)
            if mask > 0:
                fgmask = ants.morphology(fgmask,"close",mask)
            sumimg = sumimg + fgmask

    if return_sum_image:
        return avg * scl, sumimg
    if mask is None:
        avg = avg * scl
    else:
        nonzero = sumimg > sum_image_threshold
        tozero = sumimg <= sum_image_threshold
        avg[nonzero] = avg[nonzero] / sumimg[nonzero]
        avg[tozero] = 0
    return avg        

