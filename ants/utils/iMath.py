


__all__ = ['iMath',
           'image_math',
           'multiply_images',
           'iMath_get_largest_component',
           'iMath_normalize',
           'iMath_truncate_intensity',
           'iMath_sharpen',
           'iMath_pad',
           'iMath_maurer_distance',
           'iMath_perona_malik',
           'iMath_grad',
           'iMath_laplacian',
           'iMath_canny',
           'iMath_histogram_equalization',
           'iMath_MD',
           'iMath_ME',
           'iMath_MO',
           'iMath_MC',
           'iMath_GD',
           'iMath_GE',
           'iMath_GO',
           'iMath_GC',
           'iMath_fill_holes',
           'iMath_get_largest_component',
           'iMath_normalize',
           'iMath_truncate_intensity',
           'iMath_sharpen',
           'iMath_propagate_labels_through_mask']


from .process_args import _int_antsProcessArguments
from .. import utils

_iMathOps = {'FillHoles',
            'GetLargestComponent',
            'Normalize',
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


def iMath_canny(image, sigma, lower, upper):
    return iMath(image, 'Canny', sigma, lower, upper)

def iMath_distance_map(image, use_spacing=True):
    return iMath(image, 'DistanceMap', use_spacing)

def iMath_fill_holes(image, hole_type=2):
    return iMath(image, 'FillHoles', hole_type)

def iMath_GC(image, radius=1):
    return iMath(image, 'GC', radius)

def iMath_GD(image, radius=1):
    return iMath(image, 'GD', radius)

def iMath_GE(image, radius=1):
    return iMath(image, 'GE', radius)

def iMath_GO(image, radius=1):
    return iMath(image, 'GO', radius)

def iMath_get_largest_component(image, min_size=50):
    return iMath(image, 'GetLargestComponent', min_size)

def iMath_grad(image, sigma=0.5, normalize=False):
    return iMath(image, 'Grad', sigma, normalize)

def iMath_histogram_equalization(image, alpha, beta):
    return iMath(image, 'HistogramEqualization', alpha, beta)

def iMath_laplacian(image, sigma=0.5, normalize=False):
    return iMath(image, 'Laplacian', sigma, normalize)

def iMath_MC(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False):
    return iMath(image, 'MC', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_MD(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False):
    return iMath(image, 'MD', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_ME(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False):
    return iMath(image, 'ME', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_MO(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False):
    return iMath(image, 'MO', radius, value, shape, parametric, lines, thickness, include_center)

def iMath_maurer_distance(image, foreground=1):
    return iMath(image, 'MaurerDistance', foreground)

def iMath_normalize(image):
    return iMath(image, 'Normalize')

def iMath_pad(image, padding):
    return iMath(image, 'Pad', padding)

def iMath_perona_malik(image, conductance=0.25, n_iterations=1):
    return iMath(image, 'PeronaMalik', conductance, n_iterations)

def iMath_sharpen(image):
    return iMath(image, 'Sharpen')

def iMath_propagate_labels_through_mask(image, labels, stopping_value=100, propagation_method=0):
    """
    >>> import ants
    >>> wms = ants.image_read('~/desktop/wms.nii.gz')
    >>> thal = ants.image_read('~/desktop/thal.nii.gz')
    >>> img2 = ants.iMath_propagate_labels_through_mask(wms, thal, 500, 0)
    """
    return iMath(image, 'PropagateLabelsThroughMask', labels, stopping_value, propagation_method)

def iMath_truncate_intensity(image, lower_q, upper_q, n_bins=64):
    """
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> ants.iMath_truncate_intensity( img, 0.2, 0.8 )
    """
    return iMath(image, 'TruncateIntensity', lower_q, upper_q, n_bins )
