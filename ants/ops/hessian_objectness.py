__all__ = ["hessian_objectness"]

import ants
from ants.decorators import image_method
from ants.internal import get_lib_fn

@image_method
def hessian_objectness(image,
                       object_dimension=1,
                       is_bright_object=True,
                       sigma_min=0.1,
                       sigma_max=10,
                       number_of_sigma_steps=10,
                       use_sigma_logarithmic_spacing=True,
                       alpha=0.5,
                       beta=0.5,
                       gamma=5.0,
                       set_scale_objectness_measure=True):
    """

    Interface to ITK filter.  Based on the paper by Westin et al., "Geometrical 
    Diffusion Measures for MRI from Tensor Basis Analysis" and Luca Antiga's 
    Insight Journal paper http://hdl.handle.net/1926/576.

    Arguments
    ---------
    image : ANTsImage
        scalar image.

    object_dimension : unsigned int
        0: 'sphere', 1: 'line', or 2: 'plane'.

    is_bright_object : boolean
        Set 'true' for enhancing bright objects and 'false' for dark objects.

    sigma_min : float
        Define scale domain for feature extraction.

    sigma_max : float
        Define scale domain for feature extraction.

    number_of_sigma_steps : unsigned int
        Define number of samples for scale space.
        
    use_sigma_logarithmic_spacing : boolean
        Define sample spacing the for scale space.

    alpha: float 
        Hessian filter parameter.
          
    beta: float 
        Hessian filter parameter.
       
    gamma: float 
        Hessian filter parameter.

    set_scale_objectness_measure: boolean
        ...     

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> hessian_object_image = ants.hessian_objectness(image)
    """

    image_dimension = image.dimension

    libfn = get_lib_fn('hessianObjectnessF%i' % image_dimension)
    hessian = libfn(image.pointer, object_dimension, is_bright_object, sigma_min,
                                 sigma_max, number_of_sigma_steps,
                                 use_sigma_logarithmic_spacing, alpha,
                                 beta, gamma, set_scale_objectness_measure)
    output_image = ants.from_pointer(hessian).clone('float')
    return output_image
