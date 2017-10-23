
__all__ = ['quantile']

import numpy as np

def quantile(image, q, nonzero=True):
    """
    Get the quantile values from an ANTsImage
    """
    img_arr = image.numpy()
    if isinstance(q, (list,tuple)):
        q = [qq*100. if qq <= 1. else qq for qq in q]
        if nonzero:
            img_arr = img_arr[img_arr>0]
        vals = [np.percentile(img_arr, qq) for qq in q]
        return tuple(vals)
    elif isinstance(q, (float,int)):
        if q <= 1.:
            q = q*100.
        if nonzero:
            img_arr = img_arr[img_arr>0]
        return np.percentile(img_arr[img_arr>0], q)
    else:
        raise ValueError('q argument must be list/tuple or float/int')
