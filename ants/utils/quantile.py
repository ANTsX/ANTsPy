
__all__ = ['quantile']

import numpy as np

def quantile(image, q):
    """
    Get the quantile values from an ANTsImage
    """
    img_arr = image.numpy()
    if isinstance(q, (list,tuple)):
        vals = [np.percentile(img_arr, qq*100.) for qq in q]
        return tuple(vals)
    elif isinstance(q, (float,int)):
        return np.percentile(img_arr, q*100.)
    else:
        raise ValueError('q argument must be list/tuple or float/int')
