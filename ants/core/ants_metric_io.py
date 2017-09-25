"""
"""
__all__ = ['new_ants_metric',
           'create_ants_metric',
           'supported_metrics']


import os
import numpy as np

from .. import utils
from . import ants_image as iio
from . import ants_metric as mio


_supported_metrics = {'MeanSquares',
                    'MattesMutualInformation',
                    'ANTsNeighborhoodCorrelation',
                    'Correlation',
                    'Demons',
                    'JointHistogramMutualInformation'}


def supported_metrics():
    return _supported_metrics

def new_ants_metric(dimension=3, precision='float', metric_type='MeanSquares'):
    if metric_type not in _supported_metrics:
        raise ValueError('metric_type must be one of %s' % _supported_metrics)

    libfn = utils.get_lib_fn('new_ants_metricF%i'%dimension)
    itk_tx = libfn(precision, dimension, metric_type)

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

    libfn = utils.get_lib_fn('create_ants_metricF%i' % dimension)
    metric = libfn(pixeltype, dimension, metric_type, is_vector, fixed.pointer, moving.pointer)

    ants_metric = mio.ANTsImageToImageMetric(metric)

    if isinstance(fixed_mask, iio.ANTsImage):
        ants_metric.set_fixed_mask(fixed_mask)

    if isinstance(moving_mask, iio.ANTsImage):
        ants_metric.set_moving_mask(moving_mask)

    ants_metric.set_sampling(sampling_strategy, sampling_percentage)
    ants_metric.initialize()

    return ants_metric

