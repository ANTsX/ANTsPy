
 

__all__ = ['iMath']


from .process_args import _int_antsProcessArguments
from .. import lib

_iMathOps = []

def iMath(img, operation, *args):
    """
    iMath interface to perform various mathematical operations on ANTsImages
    """
    #if operation not in _iMathOps:
    #    raise ValueError('Operation not recognized')

    imgdim = img.dimension
    outimg = img.clone()
    args = [imgdim, outimg, operation, img] + [a for a in args]
    processed_args = _int_antsProcessArguments(args)
    lib.iMath(processed_args)
    return outimg



