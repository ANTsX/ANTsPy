"""
ANTs ImageToImageMetric class
"""

__all__ = []

import os
import numpy as np
from functools import partial, partialmethod
import inspect

from .. import lib
from .. import utils, registration, segmentation, viz

from . import metric_io as mio
from . import ants_image as iio


class ANTsImageToImageMetric(object):
    """
    ANTsImageToImageMetric class
    """

    def __init__(self, metric):
        self._metric = metric
        self._is_initialized = False

    # ------------------------------------------
    # PROPERTIES
    @property
    def precision(self):
        return self._metric.precision

    @property
    def dimension(self):
        return self._metric.dimension

    @property
    def metrictype(self):
        return self._metric.metrictype.replace('ImageToImageMetricv4','')

    @property
    def is_vector(self):
        return self._metric.isVector == 1

    @property
    def pointer(self):
        return self._metric.pointer

    # ------------------------------------------
    # METHODS
    def set_fixed_image(self, image):
        """
        Set Fixed ANTsImage for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setFixedImage(image._img, False)

    def set_fixed_image_mask(self, image):
        """
        Set Fixed ANTsImage Mask for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setFixedImage(image._img, True)

    def set_moving_image(self, image):
        """
        Set Moving ANTsImage for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setMovingImage(image._img, False)

    def set_moving_image_mask(self, image):
        """
        Set Fixed ANTsImage Mask for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setMovingImage(image._img, True)

    def set_sampling(self, strategy='regular', percentage=1.):
        self._metric.setSampling(strategy, percentage)

    def initialize(self):
        self._metric.initialize()
        self._is_initialized = True

    def get_value(self):
        if not self._is_initialized:
            self.initialize()
        return self._metric.getValue()

    def __repr__(self):
        s = "ANTsImageToImageMetric\n" +\
            '\t {:<10} : {}\n'.format('Dimension', self.dimension)+\
            '\t {:<10} : {}\n'.format('Precision', self.precision)+\
            '\t {:<10} : {}\n'.format('MetricType', self.metrictype)+\
            '\t {:<10} : {}\n'.format('IsVector', self.is_vector)
        return s


