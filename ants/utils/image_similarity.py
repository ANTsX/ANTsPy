

__all__ = ['image_similarity']

from ..core import ants_metric as mio
from ..core import ants_metric_io as mio2

def image_similarity(fixed_image, moving_image, metric_type='MeanSquares', 
                    fixed_mask=None, moving_mask=None, 
                    sampling_strategy='regular', sampling_percentage=1.):
    """
    Measure similarity between two images.
    NOTE: Similarity is actually returned as distance (i.e. dissimilarity)
    per ITK/ANTs convention. E.g. using Correlation metric, the similarity
    of an image with itself returns -1.
    
    ANTsR function: `imageSimilarity`

    Arguments
    ---------
    fixed : ANTsImage
        the fixed image
    
    moving : ANTsImage
        the moving image
    
    metric_type : string
        image metric to calculate
            MeanSquares
            Correlation
            ANTsNeighborhoodCorrelation
            MattesMutualInformation
            JointHistogramMutualInformation
            Demons
    
    fixed_mask : ANTsImage (optional)
        mask for the fixed image
    
    moving_mask : ANTsImage (optional)
        mask for the moving image
    
    sampling_strategy : string (optional)
        sampling strategy, default is full sampling
            None (Full sampling)
            random
            regular
    
    sampling_percentage : scalar 
        percentage of data to sample when calculating metric
        Must be between 0 and 1

    Returns
    -------
    scalar

    Example
    -------
    >>> import ants
    >>> x = ants.image_read(ants.get_ants_data('r16'))
    >>> y = ants.image_read(ants.get_ants_data('r30'))
    >>> metric = ants.image_similarity(x,y,metric_type='MeanSquares')
    """
    metric = mio2.create_ants_metric(fixed_image, moving_image, metric_type, fixed_mask,
                        moving_mask, sampling_strategy, sampling_percentage)
    return metric.get_value()

