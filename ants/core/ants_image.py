

__all__ = ['copy_image_info',
           'set_origin',
           'get_origin',
           'set_direction',
           'get_direction',
           'set_spacing',
           'get_spacing',
           'image_physical_space_consistency',
           'image_type_cast',
           'get_neighborhood_in_mask',
           'get_neighborhood_at_voxel',
           'allclose']

import os
import numpy as np
from functools import partial, partialmethod
import inspect

from .. import lib
from .. import utils, registration, segmentation, viz
from . import ants_image_io as iio2

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

_supported_ptypes = {'unsigned char', 'unsigned int', 'float', 'double'}
_short_ptype_map = {
    'unsigned char' : 'UC',
    'unsigned int': 'UI',
    'float': 'F',
    'double' : 'D'
}

# pick up lib.antsImageCloneX functions
_image_clone_dict = {}
for ndim in {2,3,4}:
    _image_clone_dict[ndim] = {}
    for d1 in _supported_ptypes:
        _image_clone_dict[ndim][d1] = {}
        for d2 in _supported_ptypes:
            d1a = _short_ptype_map[d1]
            d2a = _short_ptype_map[d2]
            try:
                _image_clone_dict[ndim][d1][d2] = 'antsImageClone%s%i%s%i'%(d1a,ndim,d2a,ndim)
            except:
                pass

# pick up lib.getNeighborhoodMatrixX functions
_get_neighborhood_matrix_dict = {}
for ndim in {2,3,4}:
    _get_neighborhood_matrix_dict[ndim] = {}
    for d1 in _supported_ptypes:
        d1a = _short_ptype_map[d1]
        try:
            _get_neighborhood_matrix_dict[ndim][d1] = 'getNeighborhoodMatrix%s%i'%(d1a,ndim)
        except:
            pass

# pick up lib.getNeighborhoodX functions
_get_neighborhood_at_voxel_dict = {}
for ndim in {2,3,4}:
    _get_neighborhood_at_voxel_dict[ndim] = {}
    for d1 in _supported_ptypes:
        d1a = _short_ptype_map[d1]
        try:
            _get_neighborhood_at_voxel_dict[ndim][d1] = 'getNeighborhood%s%i'%(d1a,ndim)
        except:
            pass



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

    def clone(self, pixeltype=None):
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
        if self.has_components:
            comp_imgs = utils.split_channels(self)
            comp_imgs_cloned = [comp_img.clone(pixeltype) for comp_img in comp_imgs]
            return utils.merge_channels(comp_imgs_cloned)
        else:
            if pixeltype is None:
                pixeltype = self.pixeltype

            ndim = self.dimension
            d1 = self.pixeltype
            d2 = pixeltype

            image_clone_fn = lib.__dict__[_image_clone_dict[ndim][d1][d2]]
            img_cloned = ANTsImage(image_clone_fn(self._img)) 
            return img_cloned

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
        if not self.has_components:
            if data.shape != self.shape:
                raise ValueError('given array shape (%s) and image array shape (%s) do not match' % (data.shape, self.shape))
        else:
            if (data.shape[0] != self.components) and (data.shape[1:] != self.shape):
                raise ValueError('given array shape (%s) and image array shape (%s) do not match' % (data.shape[1:], self.shape))

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
    def unique(self):
        """ Return unique set of values in image """
        return self.numpy().unique()

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

    def __ne__(self, other):
        this_array = self.numpy()

        if isinstance(other, ANTsImage):
            if not image_physical_space_consistency(self, other):
                raise ValueError('images do not occupy same physical space')
            other = other.numpy()

        new_array = this_array != other
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


# Set partial class methods for any functions which take an ANTsImage as the first argument
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
    target.set_spacing(reference.spacing)
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


def get_neighborhood_in_mask(image, mask, radius, physical_coordinates=False,
                            boundary_condition=None, spatial_info=False, get_gradient=False):
    """
    Get neighborhoods for voxels within mask.
    
    This converts a scalar image to a matrix with rows that contain neighbors 
    around a center voxel
    
    ANTsR function: `getNeighborhoodInMask`

    Arguments
    ---------
    image : ANTsImage
        image to get values from
    
    mask : ANTsImage
        image indicating which voxels to examine. Each voxel > 0 will be used as the 
        center of a neighborhood
    
    radius : tuple/list
        array of values for neighborhood radius (in voxels)
    
    physical_coordinates : boolean
        whether voxel indices and offsets should be in voxel or physical coordinates
    
    boundary_condition : string (optional)
        how to handle voxels in a neighborhood, but not in the mask.
            None : fill values with `NaN`
            `image` : use image value, even if not in mask
            `mean` : use mean of all non-NaN values for that neighborhood
    
    spatial_info : boolean
        whether voxel locations and neighborhood offsets should be returned along with pixel values.
    
    get_gradient : boolean
        whether a matrix of gradients (at the center voxel) should be returned in 
        addition to the value matrix (WIP)

    Returns
    -------
    if spatial_info is False:
        if get_gradient is False:
            ndarray
                an array of pixel values where the number of rows is the size of the 
                neighborhood and there is a column for each voxel

        else if get_gradient is True:
            dictionary w/ following key-value pairs:
                values : ndarray
                    array of pixel values where the number of rows is the size of the 
                    neighborhood and there is a column for each voxel.

                gradients : ndarray
                    array providing the gradients at the center voxel of each 
                    neighborhood
        
    else if spatial_info is True:
        dictionary w/ following key-value pairs:
            values : ndarray
                array of pixel values where the number of rows is the size of the 
                neighborhood and there is a column for each voxel.

            indices : ndarray
                array provinding the center coordinates for each neighborhood

            offsets : ndarray
                array providing the offsets from center for each voxel in a neighborhood

    Example
    -------
    >>> import ants
    >>> r16 = ants.image_read(ants.get_ants_data('r16'))
    >>> mask = ants.get_mask(r16)
    >>> mat = ants.get_neighborhood_in_mask(r16, mask, radius=(2,2))
    """
    if not isinstance(image, ANTsImage):
        raise ValueError('image must be ANTsImage type')
    if not isinstance(mask, ANTsImage):
        raise ValueError('mask must be ANTsImage type')
    if isinstance(radius, (int, float)):
        radius = [radius]*image.dimension
    if (not isinstance(radius, (tuple,list))) or (len(radius) != image.dimension):
        raise ValueError('radius must be tuple or list with length == image.dimension')

    boundary = 0
    if boundary_condition == 'image':
        boundary = 1
    elif boundary_condition == 'mean':
        boundary = 2
    
    get_neighborhood_matrix_fn = lib.__dict__[_get_neighborhood_matrix_dict[image.dimension][image.pixeltype]]
    retvals = get_neighborhood_matrix_fn(image._img, 
                                        mask._img, 
                                        list(radius),
                                        int(physical_coordinates), 
                                        int(boundary),
                                        int(spatial_info),
                                        int(get_gradient))
    if not spatial_info:
        if get_gradient:
            retvals['values'] = np.asarray(retvals['values'])
            retvals['gradients'] = np.asarray(retvals['gradients'])
        else:
            retvals = np.asarray(retvals['matrix'])
    else:
        retvals['values'] = np.asarray(retvals['values'])
        retvals['indices'] = np.asarray(retvals['indices'])
        retvals['offsets'] = np.asarray(retvals['offsets'])


    return retvals


def get_neighborhood_at_voxel(image, center, kernel, physical_coordinates=False):
    """
    Get a hypercube neighborhood at a voxel. Get the values in a local 
    neighborhood of an image.
    
    ANTsR function: `getNeighborhoodAtVoxel`

    Arguments
    ---------
    image : ANTsImage
        image to get values from.
    
    center : tuple/list
        indices for neighborhood center
    
    kernel : tuple/list
        either a collection of values for neighborhood radius (in voxels) or 
        a binary collection of the same dimension as the image, specifying the shape of the neighborhood to extract
    
    physical_coordinates : boolean
        whether voxel indices and offsets should be in voxel 
        or physical coordinates

    Returns
    -------
    dictionary w/ following key-value pairs:
        values : ndarray
            array of neighborhood values at the voxel

        indices : ndarray
            matrix providing the coordinates for each value

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> center = (2,2)
    >>> radius = (3,3)
    >>> retval = ants.get_neighborhood_at_voxel(img, center, radius)
    """
    if not isinstance(image, ANTsImage):
        raise ValueError('image must be ANTsImage type')

    if (not isinstance(center, (tuple,list))) or (len(center) != image.dimension):
        raise ValueError('center must be tuple or list with length == image.dimension')

    if (not isinstance(kernel, (tuple,list))) or (len(kernel) != image.dimension):
        raise ValueError('kernel must be tuple or list with length == image.dimension')

    radius = [int((k-1)/2) for k in kernel]

    get_neighborhood_at_voxel_fn = lib.__dict__[_get_neighborhood_at_voxel_dict[image.dimension][image.pixeltype]]
    retvals = get_neighborhood_at_voxel_fn(image._img, 
                                            list(center), 
                                            list(kernel), 
                                            list(radius), 
                                            int(physical_coordinates))
    for k in retvals.keys():
        retvals[k] = np.asarray(retvals[k])
    return retvals


def allclose(img, other_img):
    """
    Check if two images have the same array values
    """
    return np.allclose(img.numpy(), other_img.numpy())


