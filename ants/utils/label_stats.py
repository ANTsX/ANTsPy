
__all__ = ['label_stats']

from .. import utils

def label_stats(image, label_image):
    """
    Get label statistics from image

    ANTsR function: `labelStats`
    
    Arguments
    ---------
    image : ANTsImage 
        Image from which statistics will be calculated
    
    label_image : ANTsImage
        Label image

    Returns
    -------
    ndarray ?
    
    Example
    -------
    >>> image = ants.image_read( ants.get_ants_data('r16') , 2 )
    >>> image = ants.resample_image( image, (64,64), 1, 0 )
    >>> mask = ants.get_mask(image)
    >>> segs1 = ants.kmeans_segmentation( image, 3 )
    >>> stats = ants.label_stats(image, segs1['segmentation'])
    """
    image_float = image.clone('float')
    label_image_int = label_image.clone('unsigned int')

    libfn = utils.get_lib_fn('labelStats%iD' % image.dimension)
    df = libfn(image_float.pointer, label_image_int.pointer)
    #df = df[order(df$LabelValue), ]
    return df
