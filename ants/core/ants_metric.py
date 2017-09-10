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


class ANTsImageToImageMetric(object):
    """
    ANTsImageToImageMetric class
    """

    def __init__(self, metric):
        self._metric = metric

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
    def set_fixed_image():
        pass
    def set_moving_image():
        pass
    def set_sampling():
        pass
    def initialize():
        pass
    def get_value():
        pass

    def __repr__(self):
        s = "ANTsImageToImageMetric\n" +\
            '\t {:<10} : {}\n'.format('Dimension', self.dimension)+\
            '\t {:<10} : {}\n'.format('Precision', self.precision)+\
            '\t {:<10} : {}\n'.format('MetricType', self.metrictype)+\
            '\t {:<10} : {}\n'.format('IsVector', self.is_vector)
        return s


