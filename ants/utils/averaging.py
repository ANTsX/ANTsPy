from .. import utils, core
from ..core import ants_image_io as iio2
from .. import registration as reg
########################
def average_images( x, normalize=True, mask=None, imagetype=0 ):
    """
    average a list of images

    x : a list containing either filenames or antsImages 

    normalize : boolean

    mask : None or Otsu (string); this will perform a masked averaging which can 
        be useful when images have only partial coverage

    imagetype : integer
        choose 0/1/2/3 mapping to scalar/vector/tensor/time-series

    """
    def gli( y ):
        if isinstance(y,str):
            return iio2.image_read(y)            
        return y

    biggest=0
    biggestind=0
    for k in range( len( x ) ):
        locimg = gli( x[k] )
        sz=np.prod( locimg.shape )
        if sz > biggest:
            biggest=sz
            biggestind=k
    avg = gli( x[biggestind] ) * 0
    scl = float( 1.0 / len(x))
    if mask is not None:
        sumimg = gli( x[biggestind] ) * 0
    for k in range( len( x ) ):
        locimg = gli( x[k] )
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
    
