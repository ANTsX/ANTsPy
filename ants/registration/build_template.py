

__all__ = ['build_template']

from tempfile import mktemp

from .reflect_image import reflect_image
from .interface import registration
from .apply_transforms import apply_transforms
from ..core import ants_image_io as iio


def build_template(
    initial_template=None,
    image_list=None,
    iterations = 3,
    gradient_step = 0.2,
    **kwargs ):
    """
    Estimate an optimal template from an input image_list

    ANTsR function: N/A

    Arguments
    ---------
    initial_template : ANTsImage
        initialization for the template building

    image_list : ANTsImages
        images from which to estimate template

    iterations : integer
        number of template building iterations

    gradient_step : scalar
        for shape update gradient

    kwargs : keyword args
        extra arguments passed to ants registration

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') , 'float')
    >>> image2 = ants.image_read( ants.get_ants_data('r27') , 'float')
    >>> image3 = ants.image_read( ants.get_ants_data('r85') , 'float')
    >>> timage = ants.build_template( image_list = ( image, image2, image3 ) )
    """
    if 'type_of_transform' not in kwargs:
        type_of_transform = 'SyN'
    else:
        type_of_transform = kwargs.pop('type_of_transform')

    wt = 1.0 / len( image_list )
    if initial_template is None:
        initial_template = image_list[ 0 ] * 0
        for i in range( len( image_list ) ):
            initial_template = initial_template + image_list[ i ] * wt

    xavg = initial_template.clone()
    for i in range( iterations ):
        for k in range( len( image_list ) ):
            w1 = registration( xavg, image_list[k],
              type_of_transform=type_of_transform, **kwargs )
            if k == 0:
              wavg = iio.image_read( w1['fwdtransforms'][0] ) * wt
              xavgNew = w1['warpedmovout'] * wt
            else:
              wavg = wavg + iio.image_read( w1['fwdtransforms'][0] ) * wt
              xavgNew = xavgNew + w1['warpedmovout'] * wt
        print( wavg.abs().mean() )
        wscl = (-1.0) * gradient_step
        wavg = wavg * wscl
        wavgfn = mktemp(suffix='.nii.gz')
        iio.image_write(wavg, wavgfn)
        xavg = apply_transforms( xavgNew, xavgNew, wavgfn )

    return xavg
