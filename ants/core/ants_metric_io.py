"""
"""
__all__ = ['new_ants_metric',
           'create_ants_metric',
           'supported_metrics']


import os
import numpy as np

import ants
from ants.internal import get_lib_fn
from ants.core.ants_metric import ANTsImageToImageMetric

_supported_metrics = {'MeanSquares',
                    'MattesMutualInformation',
                    'ANTSNeighborhoodCorrelation',
                    'Correlation',
                    'Demons',
                    'JointHistogramMutualInformation'}


def supported_metrics():
    return _supported_metrics

def new_ants_metric(dimension=3, precision='float', metric_type='MeanSquares'):
    if metric_type not in _supported_metrics:
        raise ValueError('metric_type must be one of %s' % _supported_metrics)

    libfn = get_lib_fn('new_ants_metricF%i'%dimension)
    itk_tx = libfn(precision, dimension, metric_type)

    ants_metric = ANTsImageToImageMetric(itk_tx)
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
            ANTSNeighborhoodCorrelation
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
    dimension = fixed.dimension
    pixeltype = fixed.pixeltype
    is_vector = False # For now, no multichannel images

    if metric_type not in _supported_metrics:
        raise ValueError('metric_type must be one of %s' % _supported_metrics)

    if (dimension < 2) or (dimension > 4):
        raise ValueError('unsupported dimension %i' % dimension)

    if not ants.is_image(moving):
        raise ValueError('invalid moving image')

    if moving.dimension != dimension:
        raise ValueError('Fixed and Moving images must have same dimension')

    if not ants.is_image(fixed):
        raise ValueError('invalid fixed image')

    fixed = fixed.clone('float')
    moving = moving.clone('float')

    libfn = get_lib_fn('create_ants_metricF%i' % dimension)
    metric = libfn(pixeltype, dimension, metric_type, is_vector, fixed.pointer, moving.pointer)

    ants_metric = ANTsImageToImageMetric(metric)
    ants_metric.set_fixed_image(fixed)
    ants_metric.set_moving_image(moving)

    if ants.is_image(fixed_mask):
        ants_metric.set_fixed_mask(fixed_mask)

    if ants.is_image(moving_mask):
        ants_metric.set_moving_mask(moving_mask)

    ants_metric.set_sampling(sampling_strategy, sampling_percentage)
    ants_metric.initialize()

    return ants_metric

