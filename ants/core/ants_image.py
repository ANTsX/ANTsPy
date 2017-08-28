

__all__ = ['copy_image_info',
           'set_origin',
           'get_origin',
           'set_direction',
           'get_direction',
           'set_spacing',
           'get_spacing',
           'image_physical_space_consistency',
           'image_type_cast']

import os
import numpy as np
from functools import partial, partialmethod
import inspect

from .. import lib
from .. import utils, registration, segmentation, viz
from . import image_io as io

_npy_type_set = {'uint8', 'uint32', 'float32', 'float64'}
_itk_type_set = {'unsigned char', 'unsigned int', 'float', 'double'}

_itk_to_npy_map = {
    'unsigned char': 'uint8',
    'unsigned int': 'uint32',
    'float': 'float32',
    'double': 'float64'}

_npy_to_itk_map = {
    'uint8': 'unsigned char',
    'uint32':'unsigned int',
    'float32': 'float',
    'float64': 'double'}


class ANTsImage(object):

    def __init__(self, img):
        self._img = img
        self.pixeltype = img.pixeltype
        self.dtype = img.dtype
        self.dimension = img.dimension
        self.components = img.components
        self.pointer = img.pointer

    @property
    def spacing(self):
        return self._img.get_spacing()

    def set_spacing(self, new_spacing):
        self._img.set_spacing(new_spacing)

    @property
    def shape(self):
        return self._img.get_shape()

    @property
    def physical_shape(self):
        pshape = tuple([round(sh*sp,3) for sh,sp in zip(self.shape, self.spacing)])
        return pshape

    @property
    def origin(self):
        return self._img.get_origin()

    def set_origin(self, new_origin):
        self._img.set_origin(new_origin)

    @property
    def direction(self):
        return self._img.get_direction()

    def set_direction(self, new_direction):
        self._img.set_direction(new_direction)

    @property
    def has_components(self):
        return self.components > 1

    def view(self):
        dtype = self.dtype
        shape = self.shape
        if self.components > 1:
            shape = list(shape) + [self.components]
        memview = self._img.numpy()
        return np.asarray(memview).view(dtype = dtype).reshape(shape).view(np.ndarray)

    def numpy(self):
        return np.array(self.view(), copy=True)

    def clone(self, dtype=None):
        data = self.numpy()
        if dtype is not None:
            if dtype in _npy_type_set:
                data = data.astype(dtype)
            elif dtype in _itk_type_set:
                dtype = _itk_to_npy_map[dtype]
                data = data.astype(dtype)
            else:
                raise ValueError('dtype arg %s not understood' % str(dtype))

        return io.from_numpy(data=data, origin=self.origin, 
            spacing=self.spacing, direction=self.direction,
            has_components=self.has_components)

    def new_image_like(self, data):
        return io.from_numpy(data, origin=self.origin, 
            spacing=self.spacing, direction=self.direction, 
            has_components=self.has_components)

    def to_file(self, filename):
        return self._img.toFile(filename)

    def mean(self, axis=None):
        return self.numpy().mean(axis=axis)

    def std(self, axis=None):
        return self.numpy().std(axis=axis)

    def sum(self, axis=None, keepdims=False):
        return self.numpy().sum(axis=axis, keepdims=keepdims)

    def min(self, axis=None):
        return self.numpy().min(axis=axis)

    def max(self, axis=None):
        return self.numpy().max(axis=axis)

    def __add__(self, other):
        this_array = self.numpy()
        
        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array + other
        return self.new_image_like(new_array)    

    def __sub__(self, other):
        this_array = self.numpy()
        
        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array - other
        return self.new_image_like(new_array)

    def __mul__(self, other):
        this_array = self.numpy()
        
        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array * other
        return self.new_image_like(new_array)    

    def __truediv__(self, other):
        this_array = self.numpy()
        
        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array / other
        return self.new_image_like(new_array)

    def __pow__(self, other):
        this_array = self.numpy()

        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array ** other
        return self.new_image_like(new_array)

    def __gt__(self, other):
        this_array = self.numpy()

        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array > other
        return self.new_image_like(new_array.astype('uint8'))

    def __ge__(self, other):
        this_array = self.numpy()

        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array >= other
        return self.new_image_like(new_array.astype('uint8'))

    def __lt__(self, other):
        this_array = self.numpy()

        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array < other
        return self.new_image_like(new_array.astype('uint8'))

    def __le__(self, other):
        this_array = self.numpy()

        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array <= other
        return self.new_image_like(new_array.astype('uint8'))

    def __eq__(self, other):
        this_array = self.numpy()

        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array == other
        return self.new_image_like(new_array.astype('uint8'))

    def __getitem__(self, idx):
        arr = self.numpy()
        if isinstance(idx, ANTsImage):
            if not image_physical_space_consistency(self, idx):
                raise ValueError('images do not occupy same physical space')
            return arr.__getitem__(idx.numpy().astype('bool'))
        else:
            return arr.__getitem__(idx)


    def __setitem__(self, idx, value):
        arr = self.view()
        if isinstance(idx, ANTsImage):
            if not image_physical_space_consistency(self, idx):
                raise ValueError('images do not occupy same physical space')
            arr.__setitem__(idx.numpy().astype('bool'), value)
        else:
            arr.__setitem__(idx, value)

    def __repr__(self):
        s = "ANTsImage\n" +\
            '\t {:<10} : {}\n'.format('Pixel Type', self.pixeltype)+\
            '\t {:<10} : {}\n'.format('Components', self.components)+\
            '\t {:<10} : {}\n'.format('Dimensions', self.shape)+\
            '\t {:<10} : {}\n'.format('Spacing', self.spacing)+\
            '\t {:<10} : {}\n'.format('Origin', self.origin)+\
            '\t {:<10} : {}\n'.format('Direction', self.direction.flatten())
        return s

_utils_partial_dict = {k:v for k,v in utils.__dict__.items() if callable(v) and (inspect.getargspec(getattr(utils,k)).args[0] in {'img','image'})}
_reg_partial_dict = {k:v for k,v in registration.__dict__.items() if callable(v) and (inspect.getargspec(getattr(registration,k)).args[0] in {'img','image'})}
_seg_partial_dict = {k:v for k,v in segmentation.__dict__.items() if callable(v) and (inspect.getargspec(getattr(segmentation,k)).args[0] in {'img','image'})}
_viz_partial_dict = {k:v for k,v in viz.__dict__.items() if callable(v) and (inspect.getargspec(getattr(viz,k)).args[0] in {'img','image'})}

for k, v in _utils_partial_dict.items():
    setattr(ANTsImage, k, partialmethod(v))
for k, v in _reg_partial_dict.items():
    setattr(ANTsImage, k, partialmethod(v))
for k, v in _seg_partial_dict.items():
    setattr(ANTsImage, k, partialmethod(v))
for k, v in _viz_partial_dict.items():
    setattr(ANTsImage, k, partialmethod(v))



def copy_image_info(reference, target):
    """
    Copy image information from img1 to img2
    """
    target.set_origin(reference.origin)
    target.set_direction(reference.direction)
    target.set_spacing(target.spacing)
    return target

def set_origin(img, origin):
    img.set_origin(origin)

def get_origin(img):
    return img.origin

def set_direction(img, direction):
    img.set_direction(direction)

def get_direction(img):
    return img.direction

def set_spacing(img, spacing):
    img.set_spacing(spacing)

def get_spacing(img):
    return img.spacing


def image_physical_space_consistency(*imgs, tolerance=1e-2, data_type=False):
    if len(imgs) < 2:
        raise ValueError('need at least two images to compare')

    img1 = imgs[0]
    for img2 in imgs[1:]:
        if (not isinstance(img1, ANTsImage)) or (not isinstance(img2, ANTsImage)):
            raise ValueError('Both images must be of class `AntsImage`')

        # image dimension check
        if img1.dimension != img2.dimension:
            return False

        # image spacing check
        space_diffs = sum([abs(s1-s2)>tolerance for s1, s2 in zip(img1.spacing, img2.spacing)])
        if space_diffs > 0:
            return False

        # image origin check
        origin_diffs = sum([abs(s1-s2)>tolerance for s1, s2 in zip(img1.origin, img2.origin)])
        if origin_diffs > 0:
            return False

        # image direction check
        origin_diff = np.allclose(img1.direction, img2.direction, atol=tolerance)
        if not origin_diff:
            return False

        # data type
        if data_type == True:
            if img1.dtype != img2.dtype:
                return False

            if img1.n_components != img2.n_components:
                return False

    return True


def image_type_cast(image_list, pixeltype=None):
    """
    Cast a list of images to the highest pixeltype present in the list
    or all to a specified type
    """
    pixtypes = []
    for img in image_list:
        pixtypes.append(img.pixeltype)

    if pixeltype is None:
        pixeltype = 'unsigned char'
        for p in pixtypes:
            if p == 'double':
                pixeltype = 'double'
            elif (p=='float') and (pixeltype!='double'):
                pixeltype = 'float'
            elif (p=='unsigned int') and (pixeltype!='float') and (pixeltype!='double'):
                pixeltype = 'unsigned int'

    out_images = []
    for img in image_list:
        if img.pixeltype == pixeltype:
            out_images.append(img)
        else:
            out_images.append(img.clone(pixeltype))

    return out_images






