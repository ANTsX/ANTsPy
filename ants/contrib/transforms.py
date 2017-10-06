
__all__ = ['Zoom3D',
           'RandomZoom3D']

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
                 reference=None,
                 lazy=False):
        """
        Initialize a Zoom3D object

        Arguments
        ---------
        zoom_range : list or tuple
            Lower and Upper bounds on zoom parameter.
            e.g. zoom_range = (0.7,0.9) will result in a random
            draw of the zoom parameters between 0.7 and 0.9

        reference : ANTsImage (optional)
            image providing the reference space for the transform

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(zoom, (list, tuple))) or (len(zoom) != 3):
            raise ValueError('zoom_range argument must be list/tuple with three values!')

        self.zoom = zoom
        self.lazy = lazy
        self.reference = reference

        self.tx = tio.ANTsTransform(precision='float', dimension=3, 
                                    transform_type='AffineTransform')

    def transform(self, X, y=None):
        """
        Transform an image using an Affine transform with the given 
        zoom parameters

        Arguments
        ---------
        X : ANTsImage
            Image to transform

        y : ANTsImage (optional)
            Another image to transform

        Returns
        -------
        ANTsImage if y is None, else a tuple of ANTsImage types

        Examples
        --------
        >>> import ants
        >>> img = ants.image_read(ants.get_data('ch2'))
        >>> tx = ants.contrib.Zoom3D(zoom=(0.8,0.8,0.8))
        >>> img2 = tx.transform(img)
        """
        # unpack zoom range
        zoom_x, zoom_y, zoom_z = self.zoom

        self.params = (zoom_x, zoom_y, zoom_z)
        zoom_matrix = np.array([[zoom_x, 0, 0, 0],
                                [0, zoom_y, 0, 0],
                                [0, 0, zoom_z, 0]])
        if self.lazy:
            return zoom_matrix
        else:
            self.tx.set_parameters(zoom_matrix)
            return self.tx.apply_to_image(X, reference=self.reference)


class RandomZoom3D(object):
    """
    Apply a Zoom3D transform to an image, but with the zoom 
    parameters randomly generated from a user-specified range.
    """
    def __init__(self, 
                 zoom_range,
                 reference=None,
                 lazy=False):
        """
        Initialize a RandomZoom3D object

        Arguments
        ---------
        zoom_range : list or tuple
            Lower and Upper bounds on zoom parameter.
            e.g. zoom_range = (0.7,0.9) will result in a random
            draw of the zoom parameters between 0.7 and 0.9

        reference : ANTsImage (optional)
            image providing the reference space for the transform

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(zoom_range, (list, tuple))) or (len(zoom_range) != 2):
            raise ValueError('zoom_range argument must be list/tuple with two values!')

        self.zoom_range = zoom_range
        self.reference = reference
        self.lazy = lazy

    def transform(self, X, y=None):
        """
        Transform an image using an Affine transform with 
        zoom parameters randomly generated from the user-specified
        range.

        Arguments
        ---------
        X : ANTsImage
            Image to transform

        y : ANTsImage (optional)
            Another image to transform

        Returns
        -------
        ANTsImage if y is None, else a tuple of ANTsImage types

        Examples
        --------
        >>> import ants
        >>> img = ants.image_read(ants.get_data('ch2'))
        >>> tx = ants.contrib.RandomZoom3D(zoom_range=(0.8,0.9))
        >>> img2 = tx.transform(img)
        """
        # random draw in zoom range
        zoom_x = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_y = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_z = random.uniform(self.zoom_range[0], self.zoom_range[1])
        self.params = (zoom_x, zoom_y, zoom_z)
        
        tx = Zoom3D((zoom_x,zoom_y,zoom_z), 
                    reference=self.reference, 
                    lazy=self.lazy)

        return tx.transform(X,y)

