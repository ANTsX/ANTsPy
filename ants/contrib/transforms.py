

import os
import random
import math
import numpy as np

from ..core import ants_transform as tio
from ..core import ants_transform_io  as tio2

class Zoom3D(object):
    """
    Create an ANTs Affine Transform with a specified level
    of zoom. Any value greater than 1 implies a "zoom-out" and anything
    less than 1 implies a "zoom-in".
    """

    def __init__(self,
                 zoom,
                 lazy=False):
        if (not isinstance(zoom, (list, tuple))) or (len(zoom) != 3):
            raise ValueError('zoom_range argument must be list/tuple with three values!')

        self.zoom = zoom
        self.lazy = lazy

        self.tx = tio.ANTsTransform(precision='float', dimension=3, 
                                    transform_type='AffineTransform')

    def transform(self, X, y=None):
        # unpack zoom range
        zoom_x, zoom_y, zoom_z = self.zoom_tuple

        self.params = (zoom_x, zoom_y, zoom_z)
        zoom_matrix = np.array([[zoom_x, 0, 0, 0],
                                [0, zoom_y, 0, 0],
                                [0, 0, zoom_z, 0]])
        if self.lazy:
            return zoom_matrix
        else:
            self.tx.set_parameters(zoom_matrix)
            return self.tx.apply_to_image(X)

class RandomZoom3D(object):

    def __init__(self, 
                 zoom_range,
                 lazy=False):
        if (not isinstance(zoom_range, (list, tuple))) or (len(zoom_range) != 2):
            raise ValueError('zoom_range argument must be list/tuple with two values!')

        self.zoom_range = zoom_range
        self.lazy = lazy

    def transform(self, X, y=None):
        # random draw in zoom range
        zoom_x = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_y = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_z = random.uniform(self.zoom_range[0], self.zoom_range[1])
        self.params = (zoom_x, zoom_y, zoom_z)
        
        tx = Zoom3D((zoom_x,zoom_y,zoom_z), lazy=self.lazy)
        return tx.transform(X,y)

