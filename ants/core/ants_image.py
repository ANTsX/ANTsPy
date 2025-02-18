

__all__ = ['ANTsImage',
           'copy_image_info',
           'set_origin',
           'get_origin',
           'set_direction',
           'get_direction',
           'set_spacing',
           'get_spacing',
           'is_image',
           'from_pointer']

import os

import numpy as np
import pandas as pd

from functools import partialmethod
import inspect

import ants
from ants.internal import get_lib_fn

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


class ANTsImage(object):

    def __init__(self, pointer):
        """
        Initialize an ANTsImage.

        Creating an ANTsImage requires a pointer to an underlying ITK image that
        is stored via a nanobind class wrapping a AntsImage struct.

        Arguments
        ---------
        pointer : nb::class
            nanobind class wrapping the struct holding the pointer to the underlying ITK image object

        """
        self.pointer = pointer
        self.channels_first = False
        self._array = None

    @property
    def _libsuffix(self):
        return str(type(self.pointer)).split('AntsImage')[-1].split("'")[0]

    @property
    def shape(self):
        return tuple(get_lib_fn('getShape')(self.pointer))

    @property
    def physical_shape(self):
        return tuple([round(sh*sp,3) for sh,sp in zip(self.shape, self.spacing)])

    @property
    def is_rgb(self):
        return 'RGB' in self._libsuffix

    @property
    def has_components(self):
        suffix = self._libsuffix
        return suffix.startswith('V') or suffix.startswith('RGB')

    @property
    def components(self):
        if not self.has_components:
            return 1

        return get_lib_fn('getComponents')(self.pointer)

    @property
    def pixeltype(self):
        ptype = self._libsuffix[:-1]
        if self.has_components:
            if self.is_rgb:
                ptype = ptype[3:]
            else:
                ptype = ptype[1:]

        ptype_map = {'UC': 'unsigned char',
                     'UI': 'unsigned int',
                     'F': 'float',
                     'D': 'double'}
        return ptype_map[ptype]

    @property
    def dtype(self):
        return _itk_to_npy_map[self.pixeltype]

    @property
    def dimension(self):
        return int(self._libsuffix[-1])

    @property
    def spacing(self):
        """
        Get image spacing

        Returns
        -------
        tuple
        """
        return tuple(get_lib_fn('getSpacing')(self.pointer))

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

        get_lib_fn('setSpacing')(self.pointer, new_spacing)

    @property
    def origin(self):
        """
        Get image origin

        Returns
        -------
        tuple
        """
        return tuple(get_lib_fn('getOrigin')(self.pointer))

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

        get_lib_fn('setOrigin')(self.pointer, new_origin)

    @property
    def direction(self):
        """
        Get image direction

        Returns
        -------
        tuple
        """
        return np.array(get_lib_fn('getDirection')(self.pointer)).reshape(self.dimension,self.dimension)

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

        get_lib_fn('setDirection')(self.pointer, new_direction)

    @property
    def orientation(self):
        if self.dimension == 3:
            return self.get_orientation()
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
        memview = get_lib_fn('toNumpy')(img.pointer)
        return np.asarray(memview).view(dtype = dtype).reshape(shape).view(np.ndarray).T

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
        array = np.array(self.view(single_components=single_components), copy=True, dtype=self.dtype)
        if self.has_components or (single_components == True):
            if not self.channels_first:
                array = np.rollaxis(array, 0, self.dimension+1)
        return array

    def astype(self, dtype):
        """
        Cast & clone an ANTsImage to a given numpy datatype.

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

    def to_file(self, filename):
        """
        Write the ANTsImage to file

        Args
        ----
        filename : string
            filepath to which the image will be written
        """
        filename = os.path.expanduser(filename)
        get_lib_fn('toFile')(self.pointer, filename)
    to_filename = to_file

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

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array + other
        return self.new_image_like(new_array)

    __radd__ = __add__

    def __sub__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array - other
        return self.new_image_like(new_array)

    def __rsub__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = other - this_array
        return self.new_image_like(new_array)

    def __mul__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array * other
        return self.new_image_like(new_array)

    __rmul__ = __mul__

    def __truediv__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array / other
        return self.new_image_like(new_array)

    def __pow__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array ** other
        return self.new_image_like(new_array)

    def __gt__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array > other
        return self.new_image_like(new_array.astype('uint8'))

    def __ge__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array >= other
        return self.new_image_like(new_array.astype('uint8'))

    def __lt__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array < other
        return self.new_image_like(new_array.astype('uint8'))

    def __le__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array <= other
        return self.new_image_like(new_array.astype('uint8'))

    def __eq__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array == other
        return self.new_image_like(new_array.astype('uint8'))

    def __ne__(self, other):
        this_array = self.numpy()

        if is_image(other):
            if not ants.image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array != other
        return self.new_image_like(new_array.astype('uint8'))

    def __getitem__(self, idx):
        if self.has_components:
            return ants.merge_channels([
                img[idx] for img in ants.split_channels(self)
            ])

        if isinstance(idx, ANTsImage):
            if not ants.image_physical_space_consistency(self, idx):
                raise ValueError('images do not occupy same physical space')
            return self.numpy().__getitem__(idx.numpy().astype('bool'))

        # convert idx to tuple if it is not, eg im[10] or im[10:20]
        if not isinstance(idx, tuple):
            idx = (idx,)

        ndim = len(self.shape)

        if len(idx) > ndim:
            raise ValueError('Too many indices for image')
        if len(idx) < ndim:
            # If not all dimensions are indexed, assume the rest are full slices
            # eg im[10] -> im[10, :, :]
            idx = idx + (slice(None),) * (ndim - len(idx))

        sizes = list(self.shape)
        starts = [0] * ndim
        stops = list(self.shape)
        for i in range(ndim):
            ti = idx[i]
            if isinstance(ti, slice):
                if ti.start:
                    starts[i] = ti.start
                if ti.stop:
                    if ti.stop < 0:
                        stops[i] = self.shape[i] + ti.stop
                    else:
                        stops[i] = ti.stop

                sizes[i] = stops[i] - starts[i]

                if stops[i] < starts[i]:
                    raise ValueError('Reverse indexing is not supported.')

            elif isinstance(ti, int):
                starts[i] = ti
                sizes[i] = 0

            if sizes[i] == 0:
                ndim -= 1

        if ndim < 2:
            return self.numpy().__getitem__(idx)

        libfn = get_lib_fn('getItem%i' % ndim)
        new_ptr = libfn(self.pointer, starts, sizes)
        new_image = from_pointer(new_ptr)
        return new_image

    def __setitem__(self, idx, value):
        arr = self.view()

        if is_image(value):
            value = value.numpy()

        if is_image(idx):
            if not ants.image_physical_space_consistency(self, idx):
                raise ValueError('images do not occupy same physical space')
            arr.__setitem__(idx.numpy().astype('bool'), value)
        else:
            arr.__setitem__(idx, value)


    def __iter__(self):
        # Do not allow iteration on ANTsImage. Builtin iteration, eg sum(), will generally be much slower
        # than using numpy methods. We need to explicitly disallow it to prevent breaking object state.
        raise TypeError("ANTsImage is not iterable. See docs for available functions, or use numpy.")


    def __repr__(self):
        if self.dimension == 3:
            s = 'ANTsImage ({})\n'.format(self.orientation)
        else:
            s = 'ANTsImage\n'
        s = s +\
            '\t {:<10} : {} ({})\n'.format('Pixel Type', self.pixeltype, self.dtype)+\
            '\t {:<10} : {}{}\n'.format('Components', self.components, ' (RGB)' if 'RGB' in self._libsuffix else '')+\
            '\t {:<10} : {}\n'.format('Dimensions', self.shape)+\
            '\t {:<10} : {}\n'.format('Spacing', tuple([round(s,4) for s in self.spacing]))+\
            '\t {:<10} : {}\n'.format('Origin', tuple([round(o,4) for o in self.origin]))+\
            '\t {:<10} : {}\n'.format('Direction', np.round(self.direction.flatten(),4))
        return s

    def __getstate__(self):
        """
        import ants
        import pickle
        import numpy as np
        from copy import deepcopy
        img = ants.image_read( ants.get_ants_data("r16"))
        img_pickled = pickle.dumps(img)
        img2 = pickle.loads(img_pickled)
        img3 = deepcopy(img)
        img += 10
        print(img.mean(), img3.mean())
        """
        return self.numpy(), self.origin, self.spacing, self.direction, self.has_components, self.is_rgb

    def __setstate__(self, state):
        data, origin, spacing, direction, has_components, is_rgb = state
        image = ants.from_numpy(np.copy(data), origin=origin, spacing=spacing, direction=direction, has_components=has_components, is_rgb=is_rgb)
        self.__dict__ = image.__dict__

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
    target.set_spacing(reference.spacing)
    return target


def set_origin(image, origin):
    """
    Set origin of ANTsImage

    ANTsR function: `antsSetOrigin`
    """
    image.set_origin(origin)

def get_origin(image):
    """
    Get origin of ANTsImage

    ANTsR function: `antsGetOrigin`
    """

    return image.origin

def set_direction(image, direction):
    """
    Set direction of ANTsImage

    ANTsR function: `antsSetDirection`
    """
    image.set_direction(direction)

def get_direction(image):
    """
    Get direction of ANTsImage

    ANTsR function: `antsGetDirection`
    """
    return image.direction

def set_spacing(image, spacing):
    """
    Set spacing of ANTsImage

    ANTsR function: `antsSetSpacing`
    """
    image.set_spacing(spacing)

def get_spacing(image):
    """
    Get spacing of ANTsImage

    ANTsR function: `antsGetSpacing`
    """
    return image.spacing


def is_image(object):
    return isinstance(object, ANTsImage)

def from_pointer(pointer):
    return ANTsImage(pointer)