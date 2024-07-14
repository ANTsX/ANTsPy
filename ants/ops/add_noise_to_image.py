__all__ = ["add_noise_to_image"]


import ants
from ants.decorators import image_method
from ants.internal import get_lib_fn

@image_method
def add_noise_to_image(image,
                       noise_model,
                       noise_parameters
):
    """
    Add noise to an image using additive Gaussian, salt-and-pepper,
    shot, or speckle noise.

    Arguments
    ---------
    image : ANTsImage
        scalar image.

    noise_model : string
        'additivegaussian', 'saltandpepper', 'shot', or 'speckle'.

    noise_parameters : tuple or array or float
        'additivegaussian': (mean, standardDeviation)
        'saltandpepper': (probability, saltValue, pepperValue)
        'shot': scale
        'speckle': standardDeviation

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> noise_image = ants.add_noise_to_image(image, 'additivegaussian', (0.0, 1.0))
    >>> noise_image = ants.add_noise_to_image(image, 'saltandpepper', (0.1, 0.0, 100.0))
    >>> noise_image = ants.add_noise_to_image(image, 'shot', 1.0)
    >>> noise_image = ants.add_noise_to_image(image, 'speckle', 1.0)
    """

    image_dimension = image.dimension

    if noise_model == 'additivegaussian':
        if len(noise_parameters) != 2:
            raise ValueError("Incorrect number of parameters.")

        libfn = get_lib_fn("additiveGaussianNoiseF%i" % image_dimension)
        noise = libfn(image.pointer, noise_parameters[0], noise_parameters[1])
        output_image = ants.from_pointer(noise).clone('float')
        return output_image
    elif noise_model == 'saltandpepper':
        if len(noise_parameters) != 3:
            raise ValueError("Incorrect number of parameters.")

        libfn = get_lib_fn("saltAndPepperNoiseF%i" % image_dimension)
        noise = libfn(image.pointer, noise_parameters[0], noise_parameters[1], noise_parameters[2])
        output_image = ants.from_pointer(noise).clone('float')
        return output_image
    elif noise_model == 'shot':
        if not isinstance(noise_parameters, (int, float)):
            raise ValueError("Incorrect parameter specification.")

        libfn = get_lib_fn("shotNoiseF%i" % image_dimension)
        noise = libfn(image.pointer, noise_parameters)
        output_image = ants.from_pointer(noise).clone('float')
        return output_image
    elif noise_model == 'speckle':
        if not isinstance(noise_parameters, (int, float)):
            raise ValueError("Incorrect parameter specification.")

        libfn = get_lib_fn("speckleNoiseF%i" % image_dimension)
        noise = libfn(image.pointer, noise_parameters)
        output_image = ants.from_pointer(noise).clone('float')
        return output_image
    else:
        raise ValueError("Unknown noise model.")    
