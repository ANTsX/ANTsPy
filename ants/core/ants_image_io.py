"""
Image IO
"""

__all__ = ['image_header_info',
           'image_clone',
           'image_read',
           'image_write',
           'make_image',
           'matrix_to_images',
           'images_from_matrix',
           'image_list_to_matrix',
           'images_to_matrix',
           'matrix_from_images',
           'from_numpy',
           '_from_numpy']

import os
import json
import numpy as np

from . import ants_image as iio
from .. import utils
from .. import registration as reg


_supported_ptypes = {'unsigned char', 'unsigned int', 'float', 'double'}
_supported_ntypes = {'uint8', 'uint32', 'float32', 'float64'}
_unsupported_ptypes = {'char', 'unsigned short', 'short', 'int'}
_unsupported_ptype_map = {
    'char': 'float',
    'unsigned short': 'unsigned int',
    'short': 'float',
    'int': 'float',
}
_image_type_map = {
    'scalar': '',
    'vector': 'V',
    'rgb': 'RGB'
}
_ptype_type_map = {
    'unsigned char': 'UC',
    'unsigned int': 'UI',
    'float': 'F',
    'double': 'D'
}
_ntype_type_map = {
    'uint8': 'UC',
    'uint32': 'UI',
    'float32': 'F',
    'float64': 'D'
}

_image_read_dict = {}
for itype in {'scalar', 'vector', 'rgb'}:
    _image_read_dict[itype] = {}
    for p in _supported_ptypes:
        _image_read_dict[itype][p] = {}
        for d in {2,3}:
            ita = _image_type_map[itype]
            pa = _ptype_type_map[p]
            _image_read_dict[itype][p][d] = 'imageRead%s%s%i' % (ita,pa,d)

_from_numpy_dict = {}
for p in _supported_ntypes:
    _from_numpy_dict[p] = {}
    for d in {2,3}:
        ita = _image_type_map[itype]
        pa = _ntype_type_map[p]
        _from_numpy_dict[p][d] = 'fromNumpy%s%i' % (pa,d)



def from_numpy(data, origin=None, spacing=None, direction=None, has_components=False):
    """
    Create an ANTsImage object from a numpy array
    
    ANTsR function: `as.antsImage`

    Arguments
    ---------
    data : ndarray
        image data array

    origin : tuple/list
        image origin

    spacing : tuple/list
        image spacing

    direction : list/ndarray
        image direction

    has_components : boolean
        whether the image has components

    Returns
    -------
    ANTsImage
        image with given data and any given information 
    """
    return _from_numpy(data.T.copy(), origin, spacing, direction, has_components)


def _from_numpy(data, origin=None, spacing=None, direction=None, has_components=False):
    """
    Internal function for creating an ANTsImage
    """
    ndim = data.ndim
    if has_components:
        ndim -= 1
    dtype = data.dtype.name

    data = np.array(data)

    if origin is None:
        origin = tuple([0.]*ndim)
    if spacing is None:
        spacing = tuple([1.]*ndim)
    if direction is None:
        direction = np.eye(ndim)

    libfn = utils.get_lib_fn('fromNumpy%s%i' % (_ntype_type_map[dtype], ndim))

    if not has_components:
        itk_image = libfn(data, data.shape[::-1], origin, spacing, direction)
        ants_image = iio.ANTsImage(itk_image)
    else:
        arrays = [data[...,i].copy() for i in range(data.shape[-1])]
        data_shape = arrays[0].shape
        ants_images = [iio.ANTsImage(libfn(arrays[i], data_shape[::-1], origin, spacing, direction)) for i in range(len(arrays))]
        ants_image = utils.merge_channels(ants_images)

    return ants_image


def make_image(imagesize, voxval=0, spacing=None, origin=None, direction=None, has_components=False, pixeltype='float'):
    """
    Make an image with given size and voxel value or given a mask and vector
    
    ANTsR function: `makeImage`

    Arguments
    ---------
    imagesize : tuple/ANTsImage 
        input image size or mask
    
    voxval : scalar
        input image value or vector, size of mask
    
    spacing : tuple/list
        image spatial resolution
    
    origin  : tuple/list
        image spatial origin
    
    direction : list/ndarray 
        direction matrix to convert from index to physical space
    
    components : boolean 
        whether there are components per pixel or not
    
    pixeltype : float
        data type of image values

    Returns
    -------
    ANTsImage
    """
    if isinstance(imagesize, iio.ANTsImage):
        img = imagesize.clone()
        sel = imagesize > 0
        if voxval.ndim > 1:
                voxval = voxval.flatten()
        if (len(voxval) == sel.sum()) or (len(voxval) == 0):
            img[sel] = voxval
        else:
            raise ValueError('Num given voxels %i not same as num positive values %i in `imagesize`' % (len(voxval), np.sum(sel)))
        return img
    else:
        if isinstance(voxval, (tuple,list,np.ndarray)):
            voxval = np.asarray(voxval).reshape(imagesize)
        array = np.zeros(imagesize) + voxval
        image = from_numpy(array, origin=origin, spacing=spacing, direction=direction, has_components=has_components)
        return image.clone(pixeltype)


def matrix_to_images(data_matrix, mask):
    """
    Unmasks rows of a matrix and writes as images
    
    ANTsR function: `matrixToImages`

    Arguments
    ---------
    data_matrix : ndarray
        each row corresponds to an image
        array should have number of columns equal to non-zero voxels in the mask
    
    mask : ANTsImage
        image containing a binary mask. Rows of the matrix are 
        unmasked and written as images. The mask defines the output image space

    Returns
    -------
    list of ANTsImage types
    """

    if data_matrix.ndim > 2:
        data_matrix = data_matrix.reshape(data_matrix.shape[0], -1)

    numimages = len(data_matrix)
    numVoxelsInMatrix = data_matrix.shape[1]
    numVoxelsInMask = (mask>0.5).sum()
    if numVoxelsInMask != numVoxelsInMatrix:
        raise ValueError('Num masked voxels %i must match data matrix %i' % (numVoxelsInMask, numVoxelsInMatrix))

    imagelist = []
    for i in range(numimages):
        img = mask.clone()
        img[mask > 0.5] = data_matrix[i,:]
        imagelist.append(img)
    return imagelist
images_from_matrix = matrix_to_images


def images_to_matrix(image_list, mask=None, sigma=None, epsilon=0):
    """
    Read images into rows of a matrix, given a mask - much faster for 
    large datasets as it is based on C++ implementations.
    
    ANTsR function: `imagesToMatrix`

    Arguments
    ---------
    image_list : list of ANTsImage types
        images to convert to ndarray

    mask : ANTsImage (optional)
        image containing binary mask. voxels in the mask are placed in the matrix

    sigma : scaler (optional)
        smoothing factor

    epsilon : scalar
        threshold for mask

    Returns
    -------
    ndarray
        array with a row for each image
        shape = (N_IMAGES, N_VOXELS)

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data('r16'))
    >>> img2 = ants.image_read(ants.get_ants_data('r16'))
    >>> img3 = ants.image_read(ants.get_ants_data('r16'))
    >>> mat = ants.image_list_to_matrix([img,img2,img3])
    """
    def listfunc(x):
        if np.sum(np.array(x.shape) - np.array(mask.shape)) != 0:
            x = reg.resample_image_to_target(x, mask, 2)
        return x[mask]

    if mask is None:
        mask = utils.get_mask(image_list[0])

    num_images = len(image_list)
    mask_arr = mask.numpy() > epsilon
    num_voxels = np.sum(mask_arr)

    data_matrix = np.empty((num_images, num_voxels))
    do_smooth = sigma is not None
    for i,img in enumerate(image_list):
        if do_smooth:
            data_matrix[i, :] = listfunc(utils.smooth_image(img, sigma, sigma_in_physical_coordinates=True))
        else:
            data_matrix[i,:] = listfunc(img)
    return data_matrix
image_list_to_matrix = images_to_matrix
matrix_from_images = images_to_matrix


def image_header_info(filename):
    """
    Read file info from image header
    
    ANTsR function: `antsImageHeaderInfo`

    Arguments
    ---------
    filename : string
        name of image file from which info will be read

    Returns
    -------
    dict
    """
    if not os.path.exists(filename):
        raise Exception('filename does not exist')

    libfn = utils.get_lib_fn('antsImageHeaderInfo')
    retval = libfn(filename)
    return retval


def image_clone(img, pixeltype=None):
    """
    Clone an ANTsImage

    ANTsR function: `antsImageClone`
    
    Arguments
    ---------
    img : ANTsImage
        image to clone

    dtype : string (optional)
        new datatype for image

    Returns
    -------
    ANTsImage
    """
    return img.clone(pixeltype)


def image_read(filename, dimension=None, pixeltype='float'):
    """
    Read an ANTsImage from file

    ANTsR function: `antsImageRead`

    Arguments
    ---------
    filename : string
        Name of the file to read the image from.
    
    dimension : int
        Number of dimensions of the image read. This need not be the same as 
        the dimensions of the image in the file. Allowed values: 2, 3, 4. 
        If not provided, the dimension is obtained from the image file
    
    pixeltype : string
        C++ datatype to be used to represent the pixels read. This datatype 
        need not be the same as the datatype used in the file. 
        Options: unsigned char, unsigned int, float, double

    Returns
    -------
    ANTsImage
    """
    if filename.endswith('.npy'):
        filename = os.path.expanduser(filename)
        img_array = np.load(filename)
        if os.path.exists(filename.replace('.npy', '.json')):
            with open(filename.replace('.npy', '.json')) as json_data:
                img_header = json.load(json_data)
            ants_image = from_numpy(img_array,
                                    origin=img_header.get('origin', None), 
                                    spacing=img_header.get('spacing', None), 
                                    direction=np.asarray(img_header.get('direction',None)), 
                                    has_components=img_header.get('components',1)>1)
        else:
            img_header = {}
            ants_image = from_numpy(img_array)

    else:
        filename = os.path.expanduser(filename)
        if not os.path.exists(filename):
            raise ValueError('File does not exist!')

        hinfo = image_header_info(filename)
        ptype = hinfo['pixeltype']
        pclass = hinfo['pixelclass']
        ndim = hinfo['nDimensions']
        if dimension is not None:
            ndim = dimension

        if ptype in _unsupported_ptypes:
            ptype = _unsupported_ptype_map[ptype]

        libfn = utils.get_lib_fn(_image_read_dict[pclass][ptype][ndim])
        itk_pointer = libfn(filename)
        ants_image = iio.ANTsImage(pixeltype=ptype, dimension=ndim, components=1, pointer=itk_pointer)

        #if pixeltype is not None:
        #    ants_image = ants_image.clone(pixeltype)

    return ants_image


def image_write(img, filename):
    """
    Write an ANTsImage to file
    
    ANTsR function: `antsImageWrite`
    
    Arguments
    ---------
    img : ANTsImage
        image to save to file

    filename : string
        name of file to which image will be saved
    """
    if filename.endswith('.npy'):
        img_array = img.numpy()
        img_header = img.header

        np.save(filename, img_array)
        with open(filename.replace('.npy','.json'), 'w') as outfile:
            json.dump(img_header, outfile)
    else:
        img.to_file(filename)


