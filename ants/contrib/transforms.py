"""
Create transforms with specified parameters.
See http://www.cs.cornell.edu/courses/cs4620/2010fa/lectures/03transforms3d.pdf

def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], 
                              [0, 1, 0, o_y],
                              [0, 0, 1, o_z],
                              [0, 0, 0, 1]])

    reset_matrix = np.array([[1, 0, 0, -o_x], 
                             [0, 1, 0, -o_y],
                             [0, 0, 1, -o_z],
                             [0, 0, 0,  1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
"""

__all__ = ['Zoom3D',
           'RandomZoom3D',
           'Rotate3D',
           'RandomRotate3D']

import os
import random
import math
import numpy as np

from ..core import ants_transform as tio
from ..core import ants_transform_io  as tio2


class Rotate3D(object):
    """
    Create an ANTs Affine Transform with a specified level
    of rotation. 
    """
    def __init__(self,
                 rotation,
                 reference=None,
                 lazy=False):
        """
        Initialize a Rotate3D object

        Arguments
        ---------
        rotation : list or tuple
            rotation values for each axis, in degrees. 
            Negative values can be used for rotation in the 
            other direction

        reference : ANTsImage (optional)
            image providing the reference space for the transform

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(rotation, (list, tuple))) or (len(rotation) != 3):
            raise ValueError('rotation argument must be list/tuple with three values!')

        self.rotation = rotation
        self.lazy = lazy
        self.reference = reference

        self.tx = tio.ANTsTransform(precision='float', dimension=3, 
                                    transform_type='AffineTransform')

    def transform(self, X, y=None):
        """
        Transform an image using an Affine transform with the given 
        rotation parameters

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
        >>> tx = ants.contrib.Rotate3D(rotation=(10,-5,12))
        >>> img2 = tx.transform(img)
        """
        # unpack zoom range
        rotation_x, rotation_y, rotation_z = self.rotation
        
        # Rotation about X axis
        theta_x = math.pi / 180 * rotation_x
        rotate_matrix_x = np.array([[1, 0,                  0,                 0],
                                    [0, math.cos(theta_x), -math.sin(theta_x), 0],
                                    [0, math.sin(theta_x),  math.cos(theta_x), 0],
                                    [0,0,0,1]])

        # Rotation about Y axis
        theta_y = math.pi / 180 * rotation_y
        rotate_matrix_y = np.array([[math.cos(theta_y),  0, math.sin(theta_y), 0],
                                    [0,                  1, 0,                 0],
                                    [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                                    [0,0,0,1]])

        # Rotation about Z axis
        theta_z = math.pi / 180 * rotation_z
        rotate_matrix_z = np.array([[math.cos(theta_z), -math.sin(theta_z), 0, 0],
                                    [math.sin(theta_z),  math.cos(theta_z), 0, 0],
                                    [0,                0,                   1, 0],
                                    [0,0,0,1]])
        rotate_matrix = rotate_matrix_x.dot(rotate_matrix_y).dot(rotate_matrix_z)[:3,:]

        if self.lazy:
            return rotate_matrix
        else:
            self.tx.set_parameters(rotate_matrix)
            return self.tx.apply_to_image(X, reference=self.reference)


class RandomRotate3D(object):
    """
    Apply a Rotated3D transform to an image, but with the zoom 
    parameters randomly generated from a user-specified range.
    """
    def __init__(self, 
                 rotation_range,
                 reference=None,
                 lazy=False):
        """
        Initialize a RandomRotate3D object

        Arguments
        ---------
        rotation_range : list or tuple
            Lower and Upper bounds on rotation parameter, in degrees.
            e.g. rotation_range = (-10,10) will result in a random
            draw of the rotation parameters between -10 and 10 degrees

        reference : ANTsImage (optional)
            image providing the reference space for the transform

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(rotation_range, (list, tuple))) or (len(rotation_range) != 2):
            raise ValueError('rotation_range argument must be list/tuple with two values!')

        self.rotation_range = rotation_range
        self.reference = reference
        self.lazy = lazy

    def transform(self, X, y=None):
        """
        Transform an image using an Affine transform with 
        rotation parameters randomly generated from the user-specified
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
        >>> tx = ants.contrib.RandomRotate3D(rotation_range=(-10,10))
        >>> img2 = tx.transform(img)
        """
        # random draw in zoom range
        rotation_x = random.uniform(self.rotation_range[0], self.rotation_range[1])
        rotation_y = random.uniform(self.rotation_range[0], self.rotation_range[1])
        rotation_z = random.uniform(self.rotation_range[0], self.rotation_range[1])
        self.params = (rotation_x, rotation_y, rotation_z)
        
        tx = Rotate3D((rotation_x, rotation_y, rotation_z), 
                    reference=self.reference, 
                    lazy=self.lazy)

        return tx.transform(X,y)


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

