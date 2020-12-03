"""
Affine transforms

See http://www.cs.cornell.edu/courses/cs4620/2010fa/lectures/03transforms3D.pdf
"""

__all__ = [
    "Zoom2D",
    "RandomZoom2D",
    "Rotate2D",
    "RandomRotate2D",
    "Shear2D",
    "RandomShear2D",
    "Translate2D",
    "RandomTranslate2D",
]

import random
import math
import numpy as np

from ...core import ants_transform as tio


class Translate2D(object):
    """
    Create an ANTs Affine Transform with a specified translation.
    """

    def __init__(self, translation, reference=None, lazy=False):
        """
        Initialize a Shear2D object

        Arguments
        ---------
        translation : list or tuple
            translation values for each axis, in degrees.
            Negative values can be used for translation in the
            other direction

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(translation, (list, tuple))) or (len(translation) != 2):
            raise ValueError("translation argument must be list/tuple with two values!")

        self.translation = translation
        self.lazy = lazy
        self.reference = reference

        self.tx = tio.ANTsTransform(
            precision="float", dimension=2, transform_type="AffineTransform"
        )
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with the given
        translation parameters.  Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.Translate2D(translation=(10,0))
        >>> img2_x = tx.transform(img)
        >>> tx = ants.contrib.Translate2D(translation=(-10,0)) # other direction
        >>> img2_x = tx.transform(img)
        >>> tx = ants.contrib.Translate2D(translation=(0,10))
        >>> img2_z = tx.transform(img)
        >>> tx = ants.contrib.Translate2D(translation=(10,10))
        >>> img2 = tx.transform(img)
        """
        # convert to radians and unpack
        translation_x, translation_y = self.translation

        translation_matrix = np.array([[1, 0, translation_x], [0, 1, translation_y]])
        self.tx.set_parameters(translation_matrix)
        if self.lazy or X is None:
            return self.tx
        else:
            if y is None:
                return self.tx.apply_to_image(X, reference=self.reference)
            else:
                return (
                    self.tx.apply_to_image(X, reference=self.reference),
                    self.tx.apply_to_image(y, reference=self.reference),
                )


class RandomTranslate2D(object):
    """
    Apply a Translate2D transform to an image, but with the
    parameters randomly generated from a user-specified range.
    The range is determined by a mean (first parameter) and standard deviation
    (second parameter) via calls to random.gauss.
    """

    def __init__(self, translation_range, reference=None, lazy=False):
        """
        Initialize a RandomTranslate2D object

        Arguments
        ---------
        translation_range : list or tuple
            Lower and Upper bounds on rotation parameter, in degrees.
            e.g. translation_range = (-10,10) will result in a random
            draw of the rotation parameters between -10 and 10 degrees

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(translation_range, (list, tuple))) or (
            len(translation_range) != 2
        ):
            raise ValueError("shear_range argument must be list/tuple with two values!")

        self.translation_range = translation_range
        self.reference = reference
        self.lazy = lazy

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with
        translation parameters randomly generated from the user-specified
        range.   Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.RandomShear2D(translation_range=(-10,10))
        >>> img2 = tx.transform(img)
        """
        # random draw in translation range
        translation_x = random.gauss(
            self.translation_range[0], self.translation_range[1]
        )
        translation_y = random.gauss(
            self.translation_range[0], self.translation_range[1]
        )
        self.params = (translation_x, translation_y)

        tx = Translate2D(
            (translation_x, translation_y), reference=self.reference, lazy=self.lazy
        )

        return tx.transform(X, y)


class Shear2D(object):
    """
    Create an ANTs Affine Transform with a specified shear.
    """

    def __init__(self, shear, reference=None, lazy=False):
        """
        Initialize a Shear2D object

        Arguments
        ---------
        shear : list or tuple
            shear values for each axis, in degrees.
            Negative values can be used for shear in the
            other direction

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(shear, (list, tuple))) or (len(shear) != 2):
            raise ValueError("shear argument must be list/tuple with two values!")

        self.shear = shear
        self.lazy = lazy
        self.reference = reference

        self.tx = tio.ANTsTransform(
            precision="float", dimension=2, transform_type="AffineTransform"
        )
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with the given
        shear parameters.   Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.Shear2D(shear=(10,0,0))
        >>> img2_x = tx.transform(img)# x axis stays same
        >>> tx = ants.contrib.Shear2D(shear=(-10,0,0)) # other direction
        >>> img2_x = tx.transform(img)# x axis stays same
        >>> tx = ants.contrib.Shear2D(shear=(0,10,0))
        >>> img2_y = tx.transform(img) # y axis stays same
        >>> tx = ants.contrib.Shear2D(shear=(0,0,10))
        >>> img2_z = tx.transform(img) # z axis stays same
        >>> tx = ants.contrib.Shear2D(shear=(10,10,10))
        >>> img2 = tx.transform(img)
        """
        # convert to radians and unpack
        shear = [math.pi / 180 * s for s in self.shear]
        shear_x, shear_y = shear

        shear_matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0]])
        self.tx.set_parameters(shear_matrix)
        if self.lazy or X is None:
            return self.tx
        else:
            if y is None:
                return self.tx.apply_to_image(X, reference=self.reference)
            else:
                return (
                    self.tx.apply_to_image(X, reference=self.reference),
                    self.tx.apply_to_image(y, reference=self.reference),
                )


class RandomShear2D(object):
    """
    Apply a Shear2D transform to an image, but with the shear
    parameters randomly generated from a user-specified range.
    The range is determined by a mean (first parameter) and standard deviation
    (second parameter) via calls to random.gauss.
    """

    def __init__(self, shear_range, reference=None, lazy=False):
        """
        Initialize a RandomRotate2D object

        Arguments
        ---------
        shear_range : list or tuple
            Lower and Upper bounds on rotation parameter, in degrees.
            e.g. shear_range = (-10,10) will result in a random
            draw of the rotation parameters between -10 and 10 degrees

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(shear_range, (list, tuple))) or (len(shear_range) != 2):
            raise ValueError("shear_range argument must be list/tuple with two values!")

        self.shear_range = shear_range
        self.reference = reference
        self.lazy = lazy

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with
        shear parameters randomly generated from the user-specified
        range.   Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.RandomShear2D(shear_range=(-10,10))
        >>> img2 = tx.transform(img)
        """
        # random draw in shear range
        shear_x = random.gauss(self.shear_range[0], self.shear_range[1])
        shear_y = random.gauss(self.shear_range[0], self.shear_range[1])
        self.params = (shear_x, shear_y)

        tx = Shear2D((shear_x, shear_y), reference=self.reference, lazy=self.lazy)

        return tx.transform(X, y)


class Rotate2D(object):
    """
    Create an ANTs Affine Transform with a specified level
    of rotation.
    """

    def __init__(self, rotation, reference=None, lazy=False):
        """
        Initialize a Rotate2D object

        Arguments
        ---------
        rotation : scalar
            rotation value in degrees.
            Negative values can be used for rotation in the
            other direction

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        self.rotation = rotation
        self.lazy = lazy
        self.reference = reference

        self.tx = tio.ANTsTransform(
            precision="float", dimension=2, transform_type="AffineTransform"
        )
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with the given
        rotation parameters.   Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.Rotate2D(rotation=(10,-5,12))
        >>> img2 = tx.transform(img)
        """
        # unpack zoom range
        rotation = self.rotation

        # Rotation about X axis
        theta = math.pi / 180 * rotation
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]]
        )

        self.tx.set_parameters(rotation_matrix)
        if self.lazy or X is None:
            return self.tx
        else:
            if y is None:
                return self.tx.apply_to_image(X, reference=self.reference)
            else:
                return (
                    self.tx.apply_to_image(X, reference=self.reference),
                    self.tx.apply_to_image(y, reference=self.reference),
                )


class RandomRotate2D(object):
    """
    Apply a Rotated2D transform to an image, but with the zoom
    parameters randomly generated from a user-specified range.
    The range is determined by a mean (first parameter) and standard deviation
    (second parameter) via calls to random.gauss.
    """

    def __init__(self, rotation_range, reference=None, lazy=False):
        """
        Initialize a RandomRotate2D object

        Arguments
        ---------
        rotation_range : list or tuple
            Lower and Upper bounds on rotation parameter, in degrees.
            e.g. rotation_range = (-10,10) will result in a random
            draw of the rotation parameters between -10 and 10 degrees

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(rotation_range, (list, tuple))) or (
            len(rotation_range) != 2
        ):
            raise ValueError(
                "rotation_range argument must be list/tuple with two values!"
            )

        self.rotation_range = rotation_range
        self.reference = reference
        self.lazy = lazy

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with
        rotation parameters randomly generated from the user-specified
        range.   Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.RandomRotate2D(rotation_range=(-10,10))
        >>> img2 = tx.transform(img)
        """
        # random draw in rotation range
        rotation = random.gauss(self.rotation_range[0], self.rotation_range[1])
        self.params = rotation

        tx = Rotate2D(rotation, reference=self.reference, lazy=self.lazy)

        return tx.transform(X, y)


class Zoom2D(object):
    """
    Create an ANTs Affine Transform with a specified level
    of zoom. Any value greater than 1 implies a "zoom-out" and anything
    less than 1 implies a "zoom-in".
    """

    def __init__(self, zoom, reference=None, lazy=False):
        """
        Initialize a Zoom2D object

        Arguments
        ---------
        zoom_range : list or tuple
            Lower and Upper bounds on zoom parameter.
            e.g. zoom_range = (0.7,0.9) will result in a random
            draw of the zoom parameters between 0.7 and 0.9

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(zoom, (list, tuple))) or (len(zoom) != 2):
            raise ValueError("zoom_range argument must be list/tuple with two values!")

        self.zoom = zoom
        self.lazy = lazy
        self.reference = reference

        self.tx = tio.ANTsTransform(
            precision="float", dimension=2, transform_type="AffineTransform"
        )
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with the given
        zoom parameters.   Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.Zoom2D(zoom=(0.8,0.8,0.8))
        >>> img2 = tx.transform(img)
        """
        # unpack zoom range
        zoom_x, zoom_y = self.zoom

        self.params = (zoom_x, zoom_y)
        zoom_matrix = np.array([[zoom_x, 0, 0], [0, zoom_y, 0]])
        self.tx.set_parameters(zoom_matrix)
        if self.lazy or X is None:
            return self.tx
        else:
            if y is None:
                return self.tx.apply_to_image(X, reference=self.reference)
            else:
                return (
                    self.tx.apply_to_image(X, reference=self.reference),
                    self.tx.apply_to_image(y, reference=self.reference),
                )


class RandomZoom2D(object):
    """
    Apply a Zoom2D transform to an image, but with the zoom
    parameters randomly generated from a user-specified range.
    The range is determined by a mean (first parameter) and standard deviation
    (second parameter) via calls to random.gauss.
    """

    def __init__(self, zoom_range, reference=None, lazy=False):
        """
        Initialize a RandomZoom2D object

        Arguments
        ---------
        zoom_range : list or tuple
            Lower and Upper bounds on zoom parameter.
            e.g. zoom_range = (0.7,0.9) will result in a random
            draw of the zoom parameters between 0.7 and 0.9

        reference : ANTsImage (optional but recommended)
            image providing the reference space for the transform.
            this will also set the transform fixed parameters.

        lazy : boolean (default = False)
            if True, calling the `transform` method only returns
            the randomly generated transform and does not actually
            transform the image
        """
        if (not isinstance(zoom_range, (list, tuple))) or (len(zoom_range) != 2):
            raise ValueError("zoom_range argument must be list/tuple with two values!")

        self.zoom_range = zoom_range
        self.reference = reference
        self.lazy = lazy

    def transform(self, X=None, y=None):
        """
        Transform an image using an Affine transform with
        zoom parameters randomly generated from the user-specified
        range.  Return the transform if X=None.

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
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> tx = ants.contrib.RandomZoom2D(zoom_range=(0.8,0.9))
        >>> img2 = tx.transform(img)
        """
        # random draw in zoom range
        zoom_x = np.exp(
            random.gauss(np.log(self.zoom_range[0]), np.log(self.zoom_range[1]))
        )
        zoom_y = np.exp(
            random.gauss(np.log(self.zoom_range[0]), np.log(self.zoom_range[1]))
        )
        self.params = (zoom_x, zoom_y)

        tx = Zoom2D((zoom_x, zoom_y), reference=self.reference, lazy=self.lazy)

        return tx.transform(X, y)
