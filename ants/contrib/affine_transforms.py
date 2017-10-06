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

        self._params = (zoom_x, zoom_y, zoom_z)
        zoom_matrix = np.array([[zoom_x, 0, 0, 0],
                                [0, zoom_y, 0, 0],
                                [0, 0, zoom_z, 1]])
        if self.lazy:
            return zoom_matrix
        else:
            return zoom_matrix





