__all__ = ["n3_bias_field_correction", "n3_bias_field_correction2", "n4_bias_field_correction", "abp_n4"]


from . import process_args as pargs
from .get_mask import get_mask
from .iMath import iMath

from ..core import ants_image as iio
from .. import utils


def n3_bias_field_correction(image, downsample_factor=3):
    """
    N3 Bias Field Correction

    ANTsR function: `n3BiasFieldCorrection`

    Arguments
    ---------
    image : ANTsImage
        image to be bias corrected

    downsample_factor : scalar
        how much to downsample image before performing bias correction

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image_n3 = ants.n3_bias_field_correction(image)
    """
    outimage = image.clone()
    args = [image.dimension, image, outimage, downsample_factor]
    processed_args = pargs._int_antsProcessArguments(args)
    libfn = utils.get_lib_fn("N3BiasFieldCorrection")
    libfn(processed_args)
    return outimage

def n3_bias_field_correction2(
    image,
    mask=None,
    rescale_intensities=False,
    shrink_factor=4,
    convergence={"iters": 50, "tol": 1e-7},
    spline_param=200,
    number_of_fitting_levels=4,
    return_bias_field=False,
    verbose=False,
    weight_mask=None,
):
    """
    N3 Bias Field Correction

    ANTsR function: `n3BiasFieldCorrection2`

    Arguments
    ---------
    image : ANTsImage
        image to bias correct

    mask : ANTsImage
        input mask, if one is not passed one will be made

    rescale_intensities : boolean
        At each iteration, a new intensity mapping is
        calculated and applied but there is nothing which constrains the new
        intensity range to be within certain values. The result is that the
        range can "drift" from the original at each iteration. This option
        rescales to the [min,max] range of the original image intensities within
        the user-specified mask. A mask is required to perform rescaling.  Default
        is False in ANTsR/ANTsPy but True in ANTs.

    shrink_factor : scalar
        Shrink factor for multi-resolution correction, typically integer less than 4

    convergence : dict w/ keys `iters` and `tol`
        iters : maximum number of iterations
        tol : the convergence tolerance.  Default tolerance is 1e-7 in ANTsR/ANTsPy but 0.0 in ANTs.

    spline_param : float or vector Parameter controlling number of control
        points in spline. Either single value, indicating the spacing in each
        direction, or vector with one entry per dimension of image, indicating
        the mesh size.

    number_of_fitting_levels : integer
        Number of fitting levels per iteration.

    return_bias_field : boolean
        Return bias field instead of bias corrected image.

    verbose : boolean
        enables verbose output.

    weight_mask : ANTsImage (optional)
        antsImage of weight mask

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image_n3 = ants.n3_bias_field_correction2(image)
    """
    if image.pixeltype != "float":
        image = image.clone("float")
    iters = convergence["iters"]
    tol = convergence["tol"]
    if mask is None:
        mask = get_mask(image)

    N3_CONVERGENCE_1 = "[%i,%.10f]" % (iters, tol)
    N3_SHRINK_FACTOR_1 = str(shrink_factor)
    if (not isinstance(spline_param, (list, tuple))) or (len(spline_param) == 1):
        N3_BSPLINE_PARAMS = "[%i,%i]" % (spline_param, number_of_fitting_levels)
    elif (isinstance(spline_param, (list, tuple))) and (
        len(spline_param) == image.dimension
    ):
        N3_BSPLINE_PARAMS = "[%s,%i]" % (("x".join([str(sp) for sp in spline_param])), number_of_fitting_levels)
    else:
        raise ValueError(
            "Length of splineParam must either be 1 or dimensionality of image"
        )

    if weight_mask is not None:
        if not isinstance(weight_mask, iio.ANTsImage):
            raise ValueError("Weight Image must be an antsImage")

    outimage = image.clone("float")
    outbiasfield = image.clone("float")
    i = utils.get_pointer_string(outimage)
    b = utils.get_pointer_string(outbiasfield)
    output = "[%s,%s]" % (i, b)

    kwargs = {
        "d": outimage.dimension,
        "i": image,
        "w": weight_mask,
        "s": N3_SHRINK_FACTOR_1,
        "c": N3_CONVERGENCE_1,
        "b": N3_BSPLINE_PARAMS,
        "x": mask,
        "r": int(rescale_intensities),
        "o": output,
        "v": int(verbose),
    }

    processed_args = pargs._int_antsProcessArguments(kwargs)
    libfn = utils.get_lib_fn("N3BiasFieldCorrection")
    libfn(processed_args)
    if return_bias_field == True:
        return outbiasfield
    else:
        return outimage

def n4_bias_field_correction(
    image,
    mask=None,
    rescale_intensities=False,
    shrink_factor=4,
    convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
    spline_param=200,
    return_bias_field=False,
    verbose=False,
    weight_mask=None,
):
    """
    N4 Bias Field Correction

    ANTsR function: `n4BiasFieldCorrection`

    Arguments
    ---------
    image : ANTsImage
        image to bias correct

    mask : ANTsImage
        input mask, if one is not passed one will be made

    rescale_intensities : boolean
        At each iteration, a new intensity mapping is
        calculated and applied but there is nothing which constrains the new
        intensity range to be within certain values. The result is that the
        range can "drift" from the original at each iteration. This option
        rescales to the [min,max] range of the original image intensities within
        the user-specified mask. A mask is required to perform rescaling.  Default
        is False in ANTsR/ANTsPy but True in ANTs.

    shrink_factor : scalar
        Shrink factor for multi-resolution correction, typically integer less than 4

    convergence : dict w/ keys `iters` and `tol`
        iters : vector of maximum number of iterations for each level
        tol : the convergence tolerance.  Default tolerance is 1e-7 in ANTsR/ANTsPy but 0.0 in ANTs.

    spline_param : float or vector
        Parameter controlling number of control points in spline. Either single value,
        indicating the spacing in each direction, or vector with one entry per
        dimension of image, indicating the mesh size.

    return_bias_field : boolean
        Return bias field instead of bias corrected image.

    verbose : boolean
        enables verbose output.

    weight_mask : ANTsImage (optional)
        antsImage of weight mask

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image_n4 = ants.n4_bias_field_correction(image)
    """
    if image.pixeltype != "float":
        image = image.clone("float")
    iters = convergence["iters"]
    tol = convergence["tol"]
    if mask is None:
        mask = get_mask(image)

    N4_CONVERGENCE_1 = "[%s, %.10f]" % ("x".join([str(it) for it in iters]), tol)
    N4_SHRINK_FACTOR_1 = str(shrink_factor)
    if (not isinstance(spline_param, (list, tuple))) or (len(spline_param) == 1):
        N4_BSPLINE_PARAMS = "[%i]" % spline_param
    elif (isinstance(spline_param, (list, tuple))) and (
        len(spline_param) == image.dimension
    ):
        N4_BSPLINE_PARAMS = "[%s]" % ("x".join([str(sp) for sp in spline_param]))
    else:
        raise ValueError(
            "Length of splineParam must either be 1 or dimensionality of image"
        )

    if weight_mask is not None:
        if not isinstance(weight_mask, iio.ANTsImage):
            raise ValueError("Weight Image must be an antsImage")

    outimage = image.clone("float")
    outbiasfield = image.clone("float")
    i = utils.get_pointer_string(outimage)
    b = utils.get_pointer_string(outbiasfield)
    output = "[%s,%s]" % (i, b)

    kwargs = {
        "d": outimage.dimension,
        "i": image,
        "w": weight_mask,
        "s": N4_SHRINK_FACTOR_1,
        "c": N4_CONVERGENCE_1,
        "b": N4_BSPLINE_PARAMS,
        "x": mask,
        "r": int(rescale_intensities),
        "o": output,
        "v": int(verbose),
    }

    processed_args = pargs._int_antsProcessArguments(kwargs)
    libfn = utils.get_lib_fn("N4BiasFieldCorrection")
    libfn(processed_args)
    if return_bias_field == True:
        return outbiasfield
    else:
        return outimage


def abp_n4(image, intensity_truncation=(0.025, 0.975, 256), mask=None, usen3=False):
    """
    Truncate outlier intensities and bias correct with the N4 algorithm.

    ANTsR function: `abpN4`

    Arguments
    ---------
    image : ANTsImage
        image to correct and truncate

    intensity_truncation : 3-tuple
        quantiles for intensity truncation

    mask : ANTsImage (optional)
        mask for bias correction

    usen3 : boolean
        if True, use N3 bias correction instead of N4

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> image2 = ants.abp_n4(image)
    """
    if (not isinstance(intensity_truncation, (list, tuple))) or (
        len(intensity_truncation) != 3
    ):
        raise ValueError("intensity_truncation must be list/tuple with 3 values")
    outimage = iMath(
        image,
        "TruncateIntensity",
        intensity_truncation[0],
        intensity_truncation[1],
        intensity_truncation[2],
    )
    if usen3 == True:
        outimage = n3_bias_field_correction(outimage, 4)
        outimage = n3_bias_field_correction(outimage, 2)
        return outimage
    else:
        outimage = n4_bias_field_correction(outimage, mask)
        return outimage
