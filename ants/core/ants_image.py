import os

import numpy as np

from . import ants_image_utils

from .. import lib
from ..internal import short_ptype
from .ants_image_utils import image_physical_space_consistency
from . import ants_image_io as iio2

_supported_ptypes = {'unsigned char', 'unsigned int', 'float', 'double'}
_supported_dtypes = {'uint8', 'uint32', 'float32', 'float64'}
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


class AntsImage(object):

    def __init__(self, pixeltype='float', dimension=3, components=1, pointer=None, is_rgb=False):
        """
        Initialize an AntsImage

        Arguments
        ---------
        pixeltype : string
            ITK pixeltype of image

        dimension : integer
            number of image dimension. Does NOT include components dimension

        components : integer
            number of pixel components in the image

        pointer : py::capsule (optional)
            pybind11 capsule holding the pointer to the underlying ITK image object

        """
        ## Attributes which cant change without creating a new AntsImage object
        self.pointer = pointer
        self.pixeltype = pixeltype
        self.dimension = dimension
        self.components = components
        self.has_components = self.components > 1
        self.dtype = _itk_to_npy_map[self.pixeltype]
        self.is_rgb = is_rgb

        self._pixelclass = 'vector' if self.has_components else 'scalar'
        self._shortpclass = 'V' if self._pixelclass == 'vector' else ''
        if is_rgb:
            self._pixelclass = 'rgb'
            self._shortpclass = 'RGB'

        self.shape = lib.getShape(self.pointer)
        self.physical_shape = tuple([round(sh*sp,3) for sh,sp in zip(self.shape, self.spacing)])

        self._array = None

    @property
    def spacing(self):
        """
        Get image spacing

        Returns
        -------
        tuple
        """
        return lib.getSpacing(self.pointer)

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

        return lib.setSpacing(self.pointer, new_spacing)

    @property
    def origin(self):
        """
        Get image origin

        Returns
        -------
        tuple
        """
        return lib.getOrigin(self.pointer)

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

        lib.setOrigin(self.pointer, new_origin)

    @property
    def direction(self):
        """
        Get image direction

        Returns
        -------
        tuple
        """
        return np.array(lib.getDirection(self.pointer)).reshape(self.dimension, self.dimension)

    def set_direction(self, new_direction):
        """
        Set image direction

        Arguments
        ---------
        new_direction : numpy.ndarray or tuple or list
            updated direction for the image.
            should have one value for each dimension

        Returns
        -------
        None
        """
        if isinstance(new_direction, (tuple,list)):
            new_direction = np.asarray(new_direction)

        if not isinstance(new_direction, np.ndarray):
            raise ValueError('arg must be np.ndarray or tuple or list')
        if len(new_direction) != self.dimension:
            raise ValueError('must give a origin value for each dimension (%i)' % self.dimension)

        lib.setDirection(self.pointer, new_direction)

    @property
    def orientation(self):
        if self.dimension == 3:
            return get_orientation(self)
        else:
            return None

    def view(self, single_components=False):
        """
        Geet a numpy array providing direct, shared access to the image data.
        IMPORTANT: If you alter the view, then the underlying image data
        will also be altered.

        Arguments
        ---------
        single_components : boolean (default is False)
            if True, keep the extra component dimension in returned array even
            if image only has one component (i.e. self.has_components == False)

        Returns
        -------
        ndarray
        """
        if self.is_rgb:
            img = self.rgb_to_vector()
        else:
            img = self
        dtype = img.dtype
        shape = img.shape[::-1]
        if img.has_components or (single_components == True):
            shape = list(shape) + [img.components]
        
        array = np.frombuffer(lib.toNumpy(self.pointer), dtype=self.dtype).reshape(self.shape)
        if self.has_components or (single_components == True):
            array = np.rollaxis(array, 0, self.dimension+1)
        return array

    def numpy(self, single_components=False):
        """
        Get a numpy array copy representing the underlying image data. Altering
        this ndarray will have NO effect on the underlying image data.

        Arguments
        ---------
        single_components : boolean (default is False)
            if True, keep the extra component dimension in returned array even
            if image only has one component (i.e. self.has_components == False)

        Returns
        -------
        ndarray
        """
        return np.copy(self.view())

    def clone(self, pixeltype=None):
        """
        Create a copy of the given AntsImage with the same data and info, possibly with
        a different data type for the image data. Only supports casting to
        uint8 (unsigned char), uint32 (unsigned int), float32 (float), and float64 (double)

        Arguments
        ---------
        dtype: string (optional)
            if None, the dtype will be the same as the cloned AntsImage. Otherwise,
            the data will be cast to this type. This can be a numpy type or an ITK
            type.
            Options:
                'unsigned char' or 'uint8',
                'unsigned int' or 'uint32',
                'float' or 'float32',
                'double' or 'float64'

        Returns
        -------
        AntsImage
        """
        if pixeltype is None:
            pixeltype = self.pixeltype

        if pixeltype not in _supported_ptypes:
            raise ValueError('Pixeltype %s not supported. Supported types are %s' % (pixeltype, _supported_ptypes))

        if self.has_components and (not self.is_rgb):
            comp_imgs = ants_image_utils.split_channels(self)
            comp_imgs_cloned = [comp_img.clone(pixeltype) for comp_img in comp_imgs]
            return ants_image_utils.merge_channels(comp_imgs_cloned)
        else:
            p1_short = short_ptype(self.pixeltype)
            p2_short = short_ptype(pixeltype)
            ndim = self.dimension
            fn_suffix1 = '%s%i' % (p1_short,ndim)
            fn_suffix2 = '%s%i' % (p2_short,ndim)
            pointer_cloned = lib.antsImageClone(self.pointer, fn_suffix1, fn_suffix2)
            return AntsImage(pixeltype=pixeltype,
                            dimension=self.dimension,
                            components=self.components,
                            is_rgb=self.is_rgb,
                            pointer=pointer_cloned)

    # pythonic alias for `clone` is `copy`
    copy = clone

    def astype(self, dtype):
        """
        Cast & clone an AntsImage to a given numpy datatype.

        Map:
            uint8   : unsigned char
            uint32  : unsigned int
            float32 : float
            float64 : double
        """
        if dtype not in _supported_dtypes:
            raise ValueError('Datatype %s not supported. Supported types are %s' % (dtype, _supported_dtypes))

        pixeltype = _npy_to_itk_map[dtype]
        return self.clone(pixeltype)

    def new_image_like(self, data):
        """
        Create a new AntsImage with the same header information, but with
        a new image array.

        Arguments
        ---------
        data : ndarray or py::capsule
            New array or pointer for the image.
            It must have the same shape as the current
            image data.

        Returns
        -------
        AntsImage
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array')
        if not self.has_components:
            if data.shape != self.shape:
                raise ValueError('given array shape (%s) and image array shape (%s) do not match' % (data.shape, self.shape))
        else:
            if (data.shape[-1] != self.components) or (data.shape[:-1] != self.shape):
                raise ValueError('given array shape (%s) and image array shape (%s) do not match' % (data.shape[1:], self.shape))

        return iio2.from_numpy(data, origin=self.origin,
            spacing=self.spacing, direction=self.direction,
            has_components=self.has_components)

    def to_file(self, filename):
        """
        Write the AntsImage to file

        Args
        ----
        filename : string
            filepath to which the image will be written
        """
        filename = os.path.expanduser(filename)
        lib.toFile(self.pointer, filename)
    to_filename = to_file

    def apply(self, fn):
        """
        Apply an arbitrary function to AntsImage.

        Args
        ----
        fn : python function or lambda
            function to apply to ENTIRE image at once

        Returns
        -------
        AntsImage
            image with function applied to it
        """
        this_array = self.numpy()
        new_array = fn(this_array)
        return self.new_image_like(new_array)

    ## NUMPY FUNCTIONS ##
    def abs(self, axis=None):
        """ Return absolute value of image """
        return np.abs(self.numpy())
    def mean(self, axis=None):
        """ Return mean along specified axis """
        return self.numpy().mean(axis=axis)
    def median(self, axis=None):
        """ Return median along specified axis """
        return np.median(self.numpy(), axis=axis)
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
    def range(self, axis=None):
        """ Return range tuple along specified axis """
        return (self.min(axis=axis), self.max(axis=axis))
    def argmin(self, axis=None):
        """ Return argmin along specified axis """
        return self.numpy().argmin(axis=axis)
    def argmax(self, axis=None):
        """ Return argmax along specified axis """
        return self.numpy().argmax(axis=axis)
    def argrange(self, axis=None):
        """ Return argrange along specified axis """
        amin = self.argmin(axis=axis)
        amax = self.argmax(axis=axis)
        if axis is None:
            return (amin, amax)
        else:
            return np.stack([amin, amax]).T
    def flatten(self):
        """ Flatten image data """
        return self.numpy().flatten()
    def nonzero(self):
        """ Return non-zero indices of image """
        return self.numpy().nonzero()
    def unique(self, sort=False):
        """ Return unique set of values in image """
        unique_vals = np.unique(self.numpy())
        if sort:
            unique_vals = np.sort(unique_vals)
        return unique_vals

    ## OVERLOADED OPERATORS ##
    def __add__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array + other
        return self.new_image_like(new_array)
    
    __radd__ = __add__
    
    def __sub__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array - other
        return self.new_image_like(new_array)
    
    def __rsub__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = other - this_array
        return self.new_image_like(new_array)
    
    def __mul__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array * other
        return self.new_image_like(new_array)

    __rmul__ = __mul__

    def __truediv__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array / other
        return self.new_image_like(new_array)

    def __pow__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array ** other
        return self.new_image_like(new_array)

    def __gt__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array > other
        return self.new_image_like(new_array.astype('uint8'))

    def __ge__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array >= other
        return self.new_image_like(new_array.astype('uint8'))

    def __lt__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array < other
        return self.new_image_like(new_array.astype('uint8'))

    def __le__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array <= other
        return self.new_image_like(new_array.astype('uint8'))

    def __eq__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array == other
        return self.new_image_like(new_array.astype('uint8'))

    def __ne__(self, other):
        this_array = self.numpy()

        if isinstance(other, AntsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array != other
        return self.new_image_like(new_array.astype('uint8'))

    def __getitem__(self, idx):
        if self._array is None:
            self._array = self.numpy()

        if isinstance(idx, AntsImage):
            if not image_physical_space_consistency(self, idx):
                raise ValueError('images do not occupy same physical space')
            return self._array.__getitem__(idx.numpy().astype('bool'))
        else:
            return self._array.__getitem__(idx)


    def __setitem__(self, idx, value):
        arr = self.view()
        if isinstance(idx, AntsImage):
            if not image_physical_space_consistency(self, idx):
                raise ValueError('images do not occupy same physical space')
            arr.__setitem__(idx.numpy().astype('bool'), value)
        else:
            arr.__setitem__(idx, value)

    def __repr__(self):
        if self.dimension == 3:
            s = 'AntsImage ({})\n'.format(self.orientation)
        else:
            s = 'AntsImage\n'
        s = s +\
            '\t {:<10} : {} ({})\n'.format('Pixel Type', self.pixeltype, self.dtype)+\
            '\t {:<10} : {}{}\n'.format('Components', self.components, ' (RGB)' if self.is_rgb else '')+\
            '\t {:<10} : {}\n'.format('Dimensions', self.shape)+\
            '\t {:<10} : {}\n'.format('Spacing', tuple([round(s,4) for s in self.spacing]))+\
            '\t {:<10} : {}\n'.format('Origin', tuple([round(o,4) for o in self.origin]))+\
            '\t {:<10} : {}\n'.format('Direction', np.round(self.direction.flatten(),4))
        return s


def set_origin(image, origin):
    """
    Set origin of AntsImage

    ANTsR function: `antsSetOrigin`
    """
    image.set_origin(origin)

def get_origin(image):
    """
    Get origin of AntsImage

    ANTsR function: `antsGetOrigin`
    """

    return image.origin

def set_direction(image, direction):
    """
    Set direction of AntsImage

    ANTsR function: `antsSetDirection`
    """
    image.set_direction(direction)

def get_direction(image):
    """
    Get direction of AntsImage

    ANTsR function: `antsGetDirection`
    """
    return image.direction

def set_spacing(image, spacing):
    """
    Set spacing of AntsImage

    ANTsR function: `antsSetSpacing`
    """
    image.set_spacing(spacing)

def get_spacing(image):
    """
    Get spacing of AntsImage

    ANTsR function: `antsGetSpacing`
    """
    return image.spacing

def get_orientation(image):
    direction = image.direction

    orientation = []
    for i in range(3):
        row = direction[:,i]
        idx = np.where(np.abs(row)==np.max(np.abs(row)))[0][0]

        if idx == 0:
            if row[idx] < 0:
                orientation.append('L')
            else:
                orientation.append('R')
        elif idx == 1:
            if row[idx] < 0:
                orientation.append('P')
            else:
                orientation.append('A')
        elif idx == 2:
            if row[idx] < 0:
                orientation.append('S')
            else:
                orientation.append('I')
    return ''.join(orientation)
