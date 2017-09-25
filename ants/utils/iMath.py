
 

__all__ = ['iMath',
           'image_math',
           'multiply_images']


from .process_args import _int_antsProcessArguments
from .. import utils

_iMathOps = []

def iMath(image, operation, *args):
    """
    Perform various (often mathematical) operations on the input image/s. 
    Additional parameters should be specific for each operation. 
    See the the full iMath in ANTs, on which this function is based.    
    
    ANTsR function: `iMath`

    Arguments
    ---------
    image : ANTsImage
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

    imagedim = image.dimension
    outimage = image.clone()
    args = [imagedim, outimage, operation, image] + [a for a in args]
    processed_args = _int_antsProcessArguments(args)
    libfn = utils.get_lib_fn('iMath')
    libfn(processed_args)
    return outimage
image_math = iMath


def multiply_images(image1, image2):
    return image1 * image2