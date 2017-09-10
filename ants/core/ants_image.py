

__all__ = ['copy_image_info',
           'set_origin',
           'get_origin',
           'set_direction',
           'get_direction',
           'set_spacing',
           'get_spacing',
           'image_physical_space_consistency',
           'image_type_cast',
           'destroy_image']

import os
import gc
import numpy as np
from functools import partial, partialmethod
import inspect

from .. import lib
from .. import utils, registration, segmentation, viz
from . import image_io as iio2

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


def destroy_image(image):
    if image.dimension == 2:
        lib.destroy_imageF2(image._img)
    else:
        lib.destroy_imageF3(image._img)

class ANTsImage(object):

    def __init__(self, img):
        """
        NOTE: This class should never be initialized directly by the user.

        Initialize an ANTsImage

        Arguments
        ---------
        img : Cpp-ANTsImage object
            underlying cpp class which this class just wraps
        """
        self._img = img

    @property
    def pixeltype(self):
        """
        ITK pixel representation
        """
        return self._img.pixeltype

    @property
    def dtype(self):
        """
        Numpy pixel representation
        """
        return self._img.dtype

    @property
    def dimension(self):
        """
        Number of dimensions in the image
        """
        return self._img.dimension

    @property
    def components(self):
        """
        Number of components in the image
        """
        return self._img.components

    @property
    def pointer(self):
        """
        Pointer to ITK image object
        """
        return self._img.pointer

    @property
    def header(self):
        """
        Get basic image header information
        """
        return {
            'pixel type': self.pixeltype,
            'components': self.components,
            'dimensions': self.shape,
            'spacing': self.spacing,
            'origin': self.origin,
            'direction': self.direction.tolist()
        }

    @property
    def spacing(self):
        """
        Get image spacing

        Returns
        -------
        tuple
        """
        return self._img.get_spacing()

    def set_spacing(self, new_spacing):
        """
        Set image spacing

        Arguments
        ---------
        new_spacing : tuple or list
            updated spacing for the image.
            should have one value for each dimension

        Returns
        -------
        None
        """
        if not isinstance(new_spacing, (tuple, list)):
            raise ValueError('arg must be tuple or list')
        if len(new_spacing) != self.dimension:
            raise ValueError('must give a spacing value for each dimension (%i)' % self.dimension)

        self._img.set_spacing(new_spacing)

    @property
    def shape(self):
        """
        Get image shape along each axis

        Returns
        -------
        tuple
        """
        return self._img.get_shape()

    @property
    def physical_shape(self):
        """
        Get shape of image in physical space. The physical shape is the image
        array shape multiplied by the spacing

        Returns
        -------
        tuple
        """
        pshape = tuple([round(sh*sp,3) for sh,sp in zip(self.shape, self.spacing)])
        return pshape

    @property
    def origin(self):
        """
        Get image origin

        Returns
        -------
        tuple
        """
        return self._img.get_origin()

    def set_origin(self, new_origin):
        """
        Set image origin

        Arguments
        ---------
        new_origin : tuple or list
            updated origin for the image.
            should have one value for each dimension

        Returns
        -------
        None
        """
        if not isinstance(new_origin, (tuple, list)):
            raise ValueError('arg must be tuple or list')
        if len(new_origin) != self.dimension:
            raise ValueError('must give a origin value for each dimension (%i)' % self.dimension)

        self._img.set_origin(new_origin)

    @property
    def direction(self):
        """
        Get image direction

        Returns
        -------
        tuple
        """
        return self._img.get_direction()

    def set_direction(self, new_direction):
        """
        Set image direction

        Arguments
        ---------
        new_direction : tuple or list
            updated direction for the image.
            should have one value for each dimension

        Returns
        -------
        None
        """
        if not isinstance(new_direction, (tuple, list)):
            raise ValueError('arg must be tuple or list')
        if len(new_direction) != self.dimension:
            raise ValueError('must give a origin value for each dimension (%i)' % self.dimension)

        self._img.set_direction(new_direction)

    @property
    def has_components(self):
        """
        Returns true if image voxels are components

        Returns
        -------
        boolean
        """
        return self.components > 1

    def view(self):
        """
        Geet a numpy array providing direct, shared access to the image data. 
        IMPORTANT: If you alter the view, then the underlying image data 
        will also be altered.

        Returns
        -------
        ndarray
        """
        dtype = self.dtype
        shape = self.shape[::-1]
        if self.components > 1:
            shape = list(shape) + [self.components]
        memview = self._img.numpy()
        return np.asarray(memview).view(dtype = dtype).reshape(shape).view(np.ndarray).T

    def numpy(self):
        """
        Get a numpy array copy representing the underlying image data. Altering
        this ndarray will have NO effect on the underlying image data.

        Returns
        -------
        ndarray
        """
        return np.array(self.view(), copy=True)

    def clone(self, dtype=None):
        """
        Create a copy of the given ANTsImage with the same data and info, possibly with
        a different data type for the image data. Only supports casting to
        uint8 (unsigned char), uint32 (unsigned int), float32 (float), and float64 (double)

        Arguments
        ---------
        dtype: string (optional)
            if None, the dtype will be the same as the cloned ANTsImage. Otherwise,
            the data will be cast to this type. This can be a numpy type or an ITK
            type.
            Options:
                'unsigned char' or 'uint8',
                'unsigned int' or 'uint32',
                'float' or 'float32',
                'double' or 'float64'

        Returns
        -------
        ANTsImage
        """
        data = self.numpy()
        if dtype is not None:
            if dtype in _npy_type_set:
                data = data.astype(dtype)
            elif dtype in _itk_type_set:
                dtype = _itk_to_npy_map[dtype]
                data = data.astype(dtype)
            else:
                raise ValueError('dtype arg %s not understood' % str(dtype))
        
        return self.new_image_like(data)

    def new_image_like(self, data):
        """
        Create a new ANTsImage with the same header information, but with 
        a new image array.

        Arguments
        ---------
        data : ndarray
            New data for the image. Must have the same shape as the current
            image data

        Returns
        -------
        ANTsImage
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array')
        if data.shape != self.shape:
            raise ValueError('given array shape (%s) and image array shape (%s) do not match' % (data.shape, self.shape))

        return iio2.from_numpy(data, origin=self.origin, 
            spacing=self.spacing, direction=self.direction, 
            has_components=self.has_components)

    def to_file(self, filename):
        """
        Write the ANTsImage to file

        Args
        ----
        filename : string
            filepath to which the image will be written
        """
        filename = os.path.expanduser(filename)
        return self._img.toFile(filename)

    def apply(self, fn):
        """
        Apply an arbitrary function to ANTsImage.

        Args
        ----
        fn : python function or lambda
            function to apply to ENTIRE image at once

        Returns
        -------
        ANTsImage
            image with function applied to it
        """
        this_array = self.numpy()
        new_array = fn(this_array)
        return self.new_image_like(new_array)

    ## NUMPY FUNCTIONS ##
    def mean(self, axis=None):
        """ Return mean along specified axis """
        return self.numpy().mean(axis=axis)
    def median(self, axis=None):
        """ Return median along specified axis """
        return self.numpy().median(axis=axis)
    def std(self, axis=None):
        """ Return std along specified axis """
        return self.numpy().std(axis=axis)
    def sum(self, axis=None, keepdims=False):
        """ Return sum along specified axis """
        return self.numpy().sum(axis=axis, keepdims=keepdims)
    def min(self, axis=None):
        """ Return min along specified axis """
        return self.numpy().min(axis=axis)
    def max(self, axis=None):
        """ Return max along specified axis """
        return self.numpy().max(axis=axis)
    def argmin(self, axis=None):
        """ Return argmin along specified axis """
        return self.numpy().argmin(axis=self.axis)
    def argmax(self, axis=None):
        """ Return argmax along specified axis """
        return self.numpy().argmax(axis=self.axis)
    def flatten(self):
        """ Flatten image data """
        return self.numpy().flatten()
    def nonzero(self):
        """ Return non-zero indices of image """
        return self.numpy().nonzero()

    ## OVERLOADED OPERATORS ##
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
    Copy origin, direction, and spacing from one antsImage to another
    
    ANTsR function: `antsCopyImageInfo`

    Arguments
    ---------   
    reference : ANTsImage
        Image to get values from.
    target  : ANTsImAGE
        Image to copy values to

    Returns
    -------
    ANTsImage
        Target image with reference header information
    """
    target.set_origin(reference.origin)
    target.set_direction(reference.direction)
    target.set_spacing(target.spacing)
    return target

def set_origin(img, origin):
    """ 
    Set origin of ANTsImage 
    
    ANTsR function: `antsSetOrigin`
    """
    img.set_origin(origin)

def get_origin(img):
    """ 
    Get origin of ANTsImage

    ANTsR function: `antsGetOrigin`
    """

    return img.origin

def set_direction(img, direction):
    """ 
    Set direction of ANTsImage 

    ANTsR function: `antsSetDirection`
    """
    img.set_direction(direction)

def get_direction(img):
    """ 
    Get direction of ANTsImage 
    
    ANTsR function: `antsGetDirection`
    """
    return img.direction

def set_spacing(img, spacing):
    """ 
    Set spacing of ANTsImage 
    
    ANTsR function: `antsSetSpacing`
    """
    img.set_spacing(spacing)

def get_spacing(img):
    """ 
    Get spacing of ANTsImage 
    
    ANTsR function: `antsGetSpacing`
    """
    return img.spacing


def image_physical_space_consistency(*imgs, tolerance=1e-2, data_type=False):
    """
    Check if two or more ANTsImage objects occupy the same physical space
    
    ANTsR function: `antsImagePhysicalSpaceConsistency`

    Arguments
    ---------
    *imgs : ANTsImages
        images to compare

    tolerance : float
        tolerance when checking origin and spacing

    data_type : boolean
        If true, also check that the image data types are the same

    Returns
    -------
    boolean
        true if images share same physical space, false otherwise
    """
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
    
    ANTsR function: `antsImageTypeCast`

    Arguments
    ---------
    image_list : list/tuple
        images to cast

    pixeltype : string (optional)
        pixeltype to cast to. If None, images will be cast to the highest
        precision pixeltype found in image_list

    Returns
    -------
    list of ANTsImages
        given images casted to new type
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






