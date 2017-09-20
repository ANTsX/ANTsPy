
__all__ = ['label_stats']

from .. import lib

_label_stats_dict = {
    2: 'labelStats2D',
    3: 'labelStats3D',
    4: 'labelStats4D'
}

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
    >>> img = ants.image_read( ants.get_ants_data('r16') , 2 )
    >>> img = ants.resample_image( img, (64,64), 1, 0 )
    >>> mask = ants.get_mask(img)
    >>> segs1 = ants.kmeans_segmentation( img, 3 )
    >>> stats = ants.label_stats(img, segs1['segmentation'])
    """
    image_float = image.clone('float')
    label_image_int = label_image.clone('unsigned int')
    label_stats_fn = _label_stats_dict[image.dimension]

    df = lib.__dict__[label_stats_fn(image_float._img, label_image_int._img)]
    #df = df[order(df$LabelValue), ]
    return df
