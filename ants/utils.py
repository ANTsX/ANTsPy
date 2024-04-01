from . import lib
from .core import ants_image as iio

_short_ptype_map = {
    "unsigned char": "UC",
    "unsigned int": "UI",
    "float": "F",
    "double": "D",
}

def short_ptype(pixeltype):
    return _short_ptype_map[pixeltype]


def ptrstr(pointer):
    """ get string representation of a py::capsule (aka pointer) """
    return lib.ptrstr(pointer)


def process_arguments(args):
    """
    Needs to be better validated.
    """
    p_args = []
    if isinstance(args, dict):
        for argname, argval in args.items():
            if "-MULTINAME-" in argname:
                # have this little hack because python doesnt support
                # multiple dict entries w/ the same key like R lists
                argname = argname[: argname.find("-MULTINAME-")]
            if argval is not None:
                if len(argname) > 1:
                    if argname == 'verbose':
                        p_args.append('-v')
                        argval = 1
                    else:
                        p_args.append("--%s" % argname)
                else:
                    p_args.append("-%s" % argname)

                if isinstance(argval, iio.ANTsImage):
                    p_args.append(ptrstr(argval.pointer))
                elif isinstance(argval, list):
                    for av in argval:
                        if isinstance(av, iio.ANTsImage):
                            av = ptrstr(av.pointer)
                        elif str(arg) == "True":
                            av = str(1)
                        elif str(arg) == "False":
                            av = str(0)
                        p_args.append(av)
                else:
                    p_args.append(str(argval))

    elif isinstance(args, list):
        for arg in args:
            if isinstance(arg, iio.ANTsImage):
                pointer_string = ptrstr(arg.pointer)
                p_arg = pointer_string
            elif arg is None:
                pass
            elif str(arg) == "True":
                p_arg = str(1)
            elif str(arg) == "False":
                p_arg = str(0)
            else:
                p_arg = str(arg)
            p_args.append(p_arg)
    return p_args

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
        noise = lib.additiveGaussianNoise(image.pointer, f'F{image_dimension}', noise_parameters[0], noise_parameters[1])
        output_image = iio.ANTsImage(pixeltype='float', 
            dimension=image_dimension, components=1,
            pointer=noise).clone('float')
        return output_image
    elif noise_model == 'saltandpepper':
        if len(noise_parameters) != 3:
            raise ValueError("Incorrect number of parameters.")

        libfn = utils.get_lib_fn("saltAndPepperNoiseF%i" % image_dimension)
        noise = libfn(image.pointer, noise_parameters[0], noise_parameters[1], noise_parameters[2])
        output_image = iio.ANTsImage(pixeltype='float', 
            dimension=image_dimension, components=1,
            pointer=noise).clone('float')
        return output_image
    elif noise_model == 'shot':
        if not isinstance(noise_parameters, (int, float)):
            raise ValueError("Incorrect parameter specification.")

        libfn = utils.get_lib_fn("shotNoiseF%i" % image_dimension)
        noise = libfn(image.pointer, noise_parameters)
        output_image = iio.ANTsImage(pixeltype='float', 
            dimension=image_dimension, components=1,
            pointer=noise).clone('float')
        return output_image
    elif noise_model == 'speckle':
        if not isinstance(noise_parameters, (int, float)):
            raise ValueError("Incorrect parameter specification.")

        libfn = utils.get_lib_fn("speckleNoiseF%i" % image_dimension)
        noise = libfn(image.pointer, noise_parameters)
        output_image = iio.ANTsImage(pixeltype='float', 
            dimension=image_dimension, components=1,
            pointer=noise).clone('float')
        return output_image
    else:
        raise ValueError("Unknown noise model.")    
