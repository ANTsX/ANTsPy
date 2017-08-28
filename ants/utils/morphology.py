
 

__all__ = ['morphology']

from .iMath import iMath

def morphology(img, operation, radius, mtype='binary', value=1,
               shape='ball', radius_is_parametric=False, thickness=1,
               lines=3, include_center=False):
    """
    Apply morphological operations to an ANTsImage
    """
    if img.components > 1:
        raise ValueError('multichannel images not yet supported')

    _sflag_dict = {'ball': 1, 'box': 2, 'cross': 3, 'annulus': 4, 'polygon': 5}
    sFlag = _sflag_dict.get(shape, 0)

    if sFlag == 0:
        raise ValueError('invalid element shape')

    if (mtype == 'binary'):
        if (operation == 'dilate'):
            if (sFlag == 5):
                ret = iMath(img, 'MD', radius, value, sFlag, lines)
            else:
                ret = iMath(img, 'MD', radius, value, sFlag, radius_is_parametric, thickness, include_center)
        elif (operation == 'erode'):
            if (sFlag == 5):
                ret = iMath(img, 'ME', radius, value, sFlag, lines)
            else:
                ret = iMath(img, 'ME', radius, value, sFlag, radius_is_parametric, thickness, include_center)
        elif (operation == 'open'):
            if (sFlag == 5):
                ret = iMath(img, 'MO', radius, value, sFlag, lines)
            else:
                ret = iMath(img, 'MO', radius, value, sFlag, radius_is_parametric, thickness, include_center)
        elif (operation == 'close'):
            if (sFlag == 5):
                ret = iMath(img, 'MC', radius, value, sFlag, lines)
            else:
                ret = iMath(img, 'MC', radius, value, sFlag, radius_is_parametric, thickness, include_center)
        else:
            raise ValueError('Invalid morphology operation')
    elif (mtype == 'grayscale'):
        if (operation == 'dilate'):
            ret = iMath(img, 'GD', radius)
        elif (operation == 'erode'):
            ret = iMath(img, 'GE', radius)
        elif (operation == 'open'):
            ret = iMath(img, 'GO', radius)
        elif (operation == 'close'):
            ret = iMath(img, 'GC', radius)
        else:
            raise ValueError('Invalid morphology operation')
    else:
        raise ValueError('Invalid morphology type')

    return ret

