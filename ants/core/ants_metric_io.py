"""
"""
__all__ = ['new_ants_metric',
           'create_ants_metric']


import os
import numpy as np

from . import ants_image as iio
from . import ants_metric as mio
from .. import lib

_new_ants_metric_dict = {
    2: lib.new_ants_metricF2,
    3: lib.new_ants_metricF3
}

_create_ants_metric_dict = {
    2: lib.create_ants_metricF2
}

_supported_metrics = {'MeanSquares',
                    'MattesMutualInformation',
                    'ANTsNeighborhoodCorrelation',
                    'Correlation',
                    'Demons',
                    'JointHistogramMutualInformation'}


def new_ants_metric(dimension=3, precision='float', metric_type='MeanSquares'):
    if metric_type not in _supported_metrics:
        raise ValueError('metric_type must be one of %s' % _supported_metrics)

    new_ants_metric_fn = _new_ants_metric_dict[dimension]
    itk_tx = new_ants_metric_fn(precision, dimension, metric_type)

    ants_metric = mio.ANTsImageToImageMetric(itk_tx)
    return ants_metric


def create_ants_metric(fixed, 
                        moving,
                        metric_type='MeanSquares',
                        fixed_mask=None,
                        moving_mask=None,
                        sampling_strategy='regular',
                        sampling_percentage=1):
    """
    Arguments
    ---------

    metric_type : string
        which metric to use
        options:
            MeanSquares
            MattesMutualInformation
            ANTsNeighborhoodCorrelation
            Correlation
            Demons
            JointHistogramMutualInformation

    Example
    -------
    >>> import ants
    >>> fixed = ants.image_read(ants.get_ants_data('r16'))
    >>> moving = ants.image_read(ants.get_ants_data('r64'))
    >>> metric_type = 'Correlation'
    >>> metric = ants.create_ants_metric(fixed, moving, metric_type)
    """
    if metric_type not in _supported_metrics:
        raise ValueError('metric_type must be one of %s' % _supported_metrics)
    dimension = fixed.dimension
    pixeltype = 'float'

    # Check for valid dimension
    if (dimension < 2) or (dimension > 4):
        raise ValueError('unsupported dimension %i' % dimension)

    is_vector = False # For now, no multichannel images

    if isinstance(fixed, iio.ANTsImage):
        fixed = fixed.clone('float')
        dimension = fixed.dimension
        pixeltype = fixed.pixeltype
    else:
        raise ValueError('invalid fixed image')

    if isinstance(moving, iio.ANTsImage):
        moving = moving.clone('float')
        if moving.dimension != dimension:
            raise ValueError('Fixed and Moving images must have same dimension')
    else:
        raise ValueError('invalid moving image')

    create_ants_metric_fn = _create_ants_metric_dict[dimension]
    metric = create_ants_metric_fn(pixeltype, dimension, metric_type, is_vector, fixed._img, moving._img)

    ants_metric = mio.ANTsImageToImageMetric(metric)
    return ants_metric




