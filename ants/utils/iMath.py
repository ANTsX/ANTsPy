
 

__all__ = ['iMath',
           'image_math',
           'multiply_images',
           'iMath_GetLargestComponent',
           'iMath_Normalize',
           'iMath_TruncateIntensity',
           'iMath_Sharpen',
           'iMath_Pad',
           'iMath_MaurerDistance',
           'iMath_PeronaMalik',
           'iMath_Grad',
           'iMath_Laplacian',
           'iMath_Canny',
           'iMath_HistogramEqualization',
           'iMath_MD',
           'iMath_ME',
           'iMath_MO',
           'iMath_MC',
           'iMath_GD',
           'iMath_GE',
           'iMath_GO',
           'iMath_GC',
           'iMath_FillHoles',
           'iMath_GetLargestComponent',
           'iMath_Normalize',
           'iMath_TruncateIntensity',
           'iMath_Sharpen',
           'iMath_PropagateLabelsThroughMask']


from .process_args import _int_antsProcessArguments
from .. import utils

_iMathOps = {'FillHoles',
            'GetLargestComponent',
            'Normalize',
            'TruncateImageIntensity',
            'Sharpen',
            'Pad',
            'D',
            'MaurerDistance',
            'PeronaMalik',
            'Grad',
            'Laplacian',
            'Canny',
            'HistogramEqualization',
            'MD',
            'ME',
            'MO',
            'MC',
            'GD',
            'GE',
            'GO',
            'GC',
            'FillHoles',
            'GetLargestComponent',
            'LabelStats',
            'Normalize',
            'TruncateIntensity',
            'Sharpen',
            'PropagateLabelsThroughMask'}


def multiply_images(image1, image2):
    return image1 * image2


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

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> img2 = ants.iMath(img, 'Canny', 1, 5, 12)
    """
    if operation not in _iMathOps:
        raise ValueError('Operation not recognized')

    imagedim = image.dimension
    outimage = image.clone()
    args = [imagedim, outimage, operation, image] + [a for a in args]
    processed_args = _int_antsProcessArguments(args)
    libfn = utils.get_lib_fn('iMath')
    libfn(processed_args)
    return outimage
image_math = iMath


def iMath_ops():
    return _iMathOps


def iMath_Canny(image, sigma, lower, upper):
    return iMath(image, 'Canny', sigma, lower, upper)

def iMath_DistanceMap(image, use_spacing):
    return iMath(image, 'DistanceMap', use_spacing)

def iMath_FillHoles(image, hole_type):
    return iMath(image, 'FillHoles', hole_type)

def iMath_GC(image, radius):
    return iMath(image, 'GC', radius)

def iMath_GD(image, radius):
    return iMath(image, 'GD', radius)

def iMath_GE(image, radius):
    return iMath(image, 'GE', radius)

def iMath_GO(image, radius):
    return iMath(image, 'GO', radius)

def iMath_GetLargestComponent(image, min_size):
    return iMath(image, 'GetLargestComponent', min_size)

def iMath_Grad(image, sigma, normalize):
    return iMath(image, 'Grad', sigma, normalize)

def iMath_HistogramEqualization(image, alpha, beta):
    return iMath(image, 'HistogramEqualization', alpha, beta)

def iMath_Laplacian(image, sigma, normalize):
    return iMath(image, 'Laplacian', sigma, normalize)

def iMath_MC(image, radius, value, shape, parametric, lines, thickness, include_center):
    return iMath(image, 'MC', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_MD(image, radius, value, shape, parametric, lines, thickness, include_center):
    return iMath(image, 'MD', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_ME(image, radius, value, shape, parametric, lines, thickness, include_center):
    return iMath(image, 'ME', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_MO(image, radius, value, shape, parametric, lines, thickness, include_center):
    return iMath(image, 'MO', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_MaurerDistance(image, foreground):
    return iMath(image, 'MaurerDistance', foreground)

def iMath_Normalize(image):
    return iMath(image, 'Normalize')

def iMath_Pad(image, padding):
    return iMath(image, 'Pad', padding)

def iMath_PeronaMalik(image, conductance, n_iterations):
    return iMath(image, 'PeronaMalik', conductance, n_iterations)

def iMath_Sharpen(image):
    return iMath(image, 'Sharpen')

def iMath_PropagateLabelsThroughMask(image, mask, labels, stopping_value, propagation_method):
    return iMath(image, 'PropagateLabelsThroughMask', mask, labels, stopping_value, propagation_method)

def iMath_TruncateIntensity(image, n_bins, lower_q, upper_q, mask):
    return iMath(image, 'TruncateIntensity', n_bins, lower_q, upper_q, mask)









