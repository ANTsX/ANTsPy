
 

__all__ = ['iMath',
           'image_math'
           'multiply_images']


from .process_args import _int_antsProcessArguments
from .. import lib

_iMathOps = []

def iMath(img, operation, *args):
    """
    Perform various (often mathematical) operations on the input image/s. 
    Additional parameters should be specific for each operation. 
    See the the full iMath in ANTs, on which this function is based.    
    
    ANTsR function: `iMath`

    Arguments
    ---------
    img : ANTsImage
        input object, usually antsImage
    
    operation   
        a string e.g. "GetLargestComponent" ... the special case of "GetOperations" 
        or "GetOperationsFull" will return a list of operations and brief 
        description. Some operations may not be valid (WIP), but most are.
    
    *args : non-keyword arguments
        additional parameters specific to the operation
    """
    #if operation not in _iMathOps:
    #    raise ValueError('Operation not recognized')

    imgdim = img.dimension
    outimg = img.clone()
    args = [imgdim, outimg, operation, img] + [a for a in args]
    processed_args = _int_antsProcessArguments(args)
    lib.iMath(processed_args)
    return outimg
image_math = iMath


def multiply_images(img1, img2):
    return img1 * img2