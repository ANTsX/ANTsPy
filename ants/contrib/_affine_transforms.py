"""
Transformations/Augmentations to apply to ANTsImages
""" 

import os
import random
import math
import numpy as np
import scipy.ndimage as ndi


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0., channel_axis=2):
    if isinstance(fill_value, str):
        if fill_value == 'min':
            fill_value = x.min()
        elif fill_value == 'max':
            fill_value = x.max()

    # squeeze image if it's 4D (3D and a channel)
    # ergo this only supports 3D images if they have only 1 channel
    # and assumes the channel is last
    if x.ndim == 4:
        is_4d = True
        x = x[...,0]
    else:
        is_4d = False

    x = np.rollaxis(x, channel_axis, 0)
    x = x.astype('float32')

    transform = transform_matrix_offset_center(transform, x.shape[0], x.shape[1])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, 
        final_affine_matrix, final_offset, order=0, mode=fill_mode, 
        cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis+1)

    if is_4d:
        x = np.expand_dims(x, -1)
    return x


class RandomAffine(object):

    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.,
                 turn_off_frequency=None):
        """Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.

        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated between (-degrees, degrees) degrees

        translation_range : a float or a tuple/list w/ 2 floats between [0, 1)
            first value:
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)

        shear_range : float
            radian bounds on the shear transform

        zoom_range : list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
            ProTip : use 'nearest' for discrete images (e.g. segmentations)
                    and use 'constant' for continuous images

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        target_fill_mode : same as fill_mode, but for target image

        target_fill_value : same as fill_value, but for target image

        """
        self.transforms = []
        if rotation_range:
            rotation_tform = RandomRotate(rotation_range, fill_mode=fill_mode, fill_value=fill_value, lazy=True)
            self.transforms.append(rotation_tform)
            self.rtx = rotation_tform
        else:
            self.rtx = None

        if translation_range:
            translation_tform = RandomTranslate(translation_range, fill_mode=fill_mode, fill_value=fill_value, lazy=True)
            self.transforms.append(translation_tform)
            self.ttx = translation_tform
        else:
            self.ttx = None

        if shear_range:
            shear_tform = RandomShear(shear_range, fill_mode=fill_mode, fill_value=fill_value, lazy=True)
            self.transforms.append(shear_tform) 
            self.stx = shear_tform
        else:
            self.stx = None

        if zoom_range:
            zoom_tform = RandomZoom(zoom_range, fill_mode=fill_mode, fill_value=fill_value, lazy=True)
            self.transforms.append(zoom_tform)
            self.ztx = zoom_tform
        else:
            self.ztx = None

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        
        self.turn_off_frequency = turn_off_frequency
        self.frequency_counter = 0

    def transform(self, X, y=None):
        if (self.turn_off_frequency is not None) and (self.frequency_counter % self.turn_off_frequency == 0):
            tform_matrix = np.eye(3)
        else:
            # collect all of the lazily returned tform matrices
            tform_matrix = self.transforms[0].transform(X)
            for tform in self.transforms[1:]:
                tform_matrix = np.dot(tform_matrix, tform.transform(X))
        self.frequency_counter += 1 

        X = apply_transform(X, tform_matrix,
                            fill_mode=self.fill_mode, 
                            fill_value=self.fill_value)

        if y is not None:
            y = apply_transform(y, tform_matrix,
                fill_mode=self.target_fill_mode, 
                fill_value=self.target_fill_value)
            return X, y
        else:
            return X

    def get_params(self):
        vals = {}
        if self.rtx is not None:
            vals['rotation'] = self.rtx._degree
        if self.ttx is not None:
            vals['translation'] = self.ttx._txty
        if self.stx is not None:
            vals['shear'] = self.stx._shear
        if self.ztx is not None:
            vals['zoom'] = self.ztx._zoom
        return vals


class AffineCompose(object):

    def __init__(self, 
                 transforms, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        """Apply a collection of explicit affine transforms to an input image,
        and to a target image if necessary

        Arguments
        ---------
        transforms : list or tuple
            each element in the list/tuple should be an affine transform.
            currently supported transforms:
                - Rotate()
                - Translate()
                - Shear()
                - Zoom()

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        """
        self.transforms = transforms
        # set transforms to lazy so they only return the tform matrix
        for t in self.transforms:
            t.lazy = True
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def transform(self, X, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0].transform(X)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform.transform(X)) 

        X = apply_transform(X, tform_matrix,
                            fill_mode=self.fill_mode, 
                            fill_value=self.fill_value)

        if y is not None:
            y = apply_transform(y, tform_matrix,
                                fill_mode=self.target_fill_mode, 
                                fill_value=self.target_fill_value)
            return X, y
        else:
            return X


class RandomRotate(object):

    def __init__(self, 
                 rotation_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : tuple of integers or integer
            image will be rotated between (rotation_range[0], rotation_range[1]) degrees
            if rotation_range is tuple, else (-rotation_range, rotation_range) if scalar

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if not isinstance(rotation_range, (tuple,list,np.ndarray)):
            rotation_range = (rotation_range, rotation_range)
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def transform(self, X, y=None):
        degree = random.uniform(self.rotation_range[0], self.rotation_range[1])
        self._degree = degree
        theta = math.pi / 180 * degree
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            x_transformed = apply_transform(X, rotation_matrix, 
                                            fill_mode=self.fill_mode, 
                                            fill_value=self.fill_value)
            if y is not None:
                y_transformed = apply_transform(y, rotation_matrix,
                                                fill_mode=self.target_fill_mode, 
                                                fill_value=self.target_fill_value)
                return x_transformed, y_transformed
            else:
                return x_transformed


class RandomTranslate(object):

    def __init__(self, 
                 translation_range, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly translate an image some fraction of total height and/or
        some fraction of total width. If the image has multiple channels,
        the same translation will be applied to each channel.

        Arguments
        ---------
        translation_range : two floats between [0, 1) 
            first value:
                fractional bounds of total height to shift image
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                fractional bounds of total width to shift image 
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def transform(self, X, y=None):
        # height shift
        if self.height_range > 0:
            tx = random.uniform(-self.height_range, self.height_range) * X.shape[0]
        else:
            tx = 0
        # width shift
        if self.width_range > 0:
            ty = random.uniform(-self.width_range, self.width_range) * X.shape[1]
        else:
            ty = 0

        self._txty = (tx,ty)
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            x_transformed = apply_transform(X, translation_matrix, 
                                            fill_mode=self.fill_mode, 
                                            fill_value=self.fill_value)
            if y is not None:
                y_transformed = apply_transform(y, translation_matrix,
                                                fill_mode=self.target_fill_mode, 
                                                fill_value=self.target_fill_value)
                return x_transformed, y_transformed
            else:
                return x_transformed


class RandomShear(object):

    def __init__(self, 
                 shear_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly shear an image with radians (-shear_range, shear_range)

        Arguments
        ---------
        shear_range : float
            radian bounds on the shear transform
        
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def transform(self, X, y=None):
        shear = random.uniform(self.shear_range[0], self.shear_range[1])
        self._shear = shear
        shear = (math.pi * shear) / 180
        
        shear_matrix = np.array([[1, -math.sin(shear), 0],
                                 [0, math.cos(shear), 0],
                                 [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            x_transformed = apply_transform(X, shear_matrix, 
                                            fill_mode=self.fill_mode, 
                                            fill_value=self.fill_value)
            if y is not None:
                y_transformed = apply_transform(y, shear_matrix,
                                                fill_mode=self.target_fill_mode, 
                                                fill_value=self.target_fill_value)
                return x_transformed, y_transformed
            else:
                return x_transformed


class RandomZoom(object):

    def __init__(self, 
                 zoom_range, 
                 fill_mode='constant', 
                 fill_value=0, 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly zoom in and/or out on an image 

        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def transform(self, X, y=None):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        self._zoom = (zx,zy)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if self.lazy:
            return zoom_matrix
        else:
            x_transformed = apply_transform(X, zoom_matrix, 
                                            fill_mode=self.fill_mode, 
                                            fill_value=self.fill_value)
            if y is not None:
                y_transformed = apply_transform(y, zoom_matrix,
                                                fill_mode=self.target_fill_mode, 
                                                fill_value=self.target_fill_value)
                return x_transformed, y_transformed
            else:
                return x_transformed



