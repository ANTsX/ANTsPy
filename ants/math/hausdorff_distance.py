__all__ = ["hausdorff_distance"]

from .. import utils


def hausdorff_distance(image1, image2):
    """
    Get Hausdorff distance between non-zero pixels in two images

    ANTsR function: `hausdorffDistance`

    Arguments
    ---------
    source image : ANTsImage
        Source image

    target_image : ANTsImage
        Target image

    Returns
    -------
    data frame with "Distance" and "AverageDistance"

    Example
    -------
    >>> import ants
    >>> r16 = ants.image_read( ants.get_ants_data('r16') )
    >>> r64 = ants.image_read( ants.get_ants_data('r64') )
    >>> s16 = ants.kmeans_segmentation( r16, 3 )['segmentation']
    >>> s64 = ants.kmeans_segmentation( r64, 3 )['segmentation']
    >>> stats = ants.hausdorff_distance(s16, s64)
    """
    image1_int = image1.clone("unsigned int")
    image2_int = image2.clone("unsigned int")

    libfn = utils.get_lib_fn("hausdorffDistance%iD" % image1_int.dimension)
    d = libfn(image1_int.pointer, image2_int.pointer)

    return d
