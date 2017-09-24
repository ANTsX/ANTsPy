
 

__all__ = ['image_mutual_information']


from .. import utils


def image_mutual_information(img1, img2):
    """
    Compute mutual information between two ANTsImage types

    ANTsR function: `antsImageMutualInformation`
    
    Arguments
    ---------
    img1 : ANTsImage
        image 1

    img2 : ANTsImage
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
    if (img1.pixeltype != 'float') or (img2.pixeltype != 'float'):
        raise ValueError('Both images must have float pixeltype')

    if img1.dimension != img2.dimension:
        raise ValueError('Both images must have same dimension')

    libfn = utils.get_lib_fn('antsImageMutualInformation%iD' % img1.dimension)
    return libfn(img1.pointer, img2.pointer)
