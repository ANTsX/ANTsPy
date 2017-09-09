
#__all__ = ['create_ants_image_to_image_metric',
#           'new_ants_image_to_image_metric',
#           'read_image_to_image_metric',
#           'write_image_to_image_metric']

import os
import numpy as np

from . import ants_image_to_image_metric as mio
from .. import lib

_new_ants_metric_dict = {
    'float': {
        2: lib.new_ants_metricF2,
        3: lib.new_ants_metricF3,
        4: lib.new_ants_metricF4
    },
    'double': {
        2: lib.new_ants_metricD2,
        3: lib.new_ants_metricD3,
        4: lib.new_ants_metricD4
    }
}

def new_image_to_image_metric(dimension=3, precision='float', metric_type='MeanSquares'):
    new_ants_metric_fn = _new_ants_metric_dict[precision][dimension]

    itk_tx = new_ants_metric_fn(precision, dimension, metric_type)
    ants_metric = mio.ants_metric(itk_tx)

    return ants_tx