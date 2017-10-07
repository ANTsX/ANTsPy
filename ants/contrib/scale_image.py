from .. import utils
from .. import core

def scale_image2d(image, scale1, scale2):
    """
    Scale an 2D image by a specified factor in each dimension.

    Arguments
    ---------
    image : ANTsImage type
        image to be scaled

    scale1 : float
        scale factor in x direction

    scale 2 : float
        scale factor in y direction

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> img_scaled = ants.scale_image2d(img, 0.8, 0.8)
    >>> ants.plot(img)
    >>> ants.plot(img_scaled)
    """
    image = image.clone('float')

    # get function from its name
    lib_fn = utils.get_lib_fn('scaleImage2D')

    # apply the function to my image
    # remember this function returns a py::capsule
    scaled_img_ptr = lib_fn(image.pointer, scale1, scale2)

    # wrap the py::capsule back in ANTsImage class
    scaled_img = core.ANTsImage(pixeltype=image.pixeltype,
                                dimension=image.dimension,
                                components=image.components,
                                pointer=scaled_img_ptr)
    return scaled_img