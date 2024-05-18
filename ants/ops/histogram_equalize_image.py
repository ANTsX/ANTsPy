__all__ = ['histogram_equalize_image']

import numpy as np

import ants
from ants.decorators import image_method

@image_method
def histogram_equalize_image(image, number_of_histogram_bins=256):
    """
    Histogram equalize image

    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html      
    
    Arguments
    ---------
    image : ANTsImage
        source image

    number_of_histogram_bins : integer
        number of bins for cumulative histogram

    Example
    -------
    >>> import ants
    >>> src_img = ants.image_read(ants.get_data('r16'))
    >>> src_img_eq = ants.histogram_equalize_image(src_img)
    """

    image_array = image.numpy()
    image_histogram, bins = np.histogram(image_array.flatten(), number_of_histogram_bins, density=True)
    cdf = image_histogram.cumsum()
    cdf = image_array.max() * cdf / cdf[-1]
    image_array_equalized_flat = np.interp(image_array.flatten(), bins[:-1], cdf)
    image_array_equalized = image_array_equalized_flat.reshape(image_array.shape)    
    image_array_equalized = ( ( image_array_equalized - image_array_equalized.min() ) /
                              ( image_array_equalized.max() - image_array_equalized.min() ) )
    image_array_equalized = image_array_equalized * ( image_array.max() - image_array.min() ) + image_array.min()

    image_equalized = ants.from_numpy(image_array_equalized)

    return ants.copy_image_info(image, image_equalized)


