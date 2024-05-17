
 

__all__ = ['image_mutual_information']


from .. import utils


def image_mutual_information(image1, image2):
    """
    Compute mutual information between two ANTsImage types

    ANTsR function: `antsImageMutualInformation`
    
    Arguments
    ---------
    image1 : ANTsImage
        image 1

    image2 : ANTsImage
        image 2

    Returns
    -------
    scalar

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16') ).clone('float')
    >>> mi = ants.image_read( ants.get_ants_data('r64') ).clone('float')
    >>> mival = ants.image_mutual_information(fi, mi) # -0.1796141
    """
    if (image1.pixeltype != 'float') or (image2.pixeltype != 'float'):
        raise ValueError('Both images must have float pixeltype')

    if image1.dimension != image2.dimension:
        raise ValueError('Both images must have same dimension')

    libfn = utils.get_lib_fn('antsImageMutualInformation%iD' % image1.dimension)
    return libfn(image1.pointer, image2.pointer)
