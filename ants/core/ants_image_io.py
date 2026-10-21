"""
Image IO
"""

__all__ = [
    "image_header_info",
    "image_clone",
    "image_read",
    "dicom_read",
    "image_write",
    "make_image",
    "from_numpy",
    "from_numpy_like",
    "new_image_like",
    "read_image_metadata",
]

import os
import json
import numpy as np

import ants
from ants.internal import get_lib_fn
from ants.decorators import image_method
from ._pixel_types import (
    _get_best_supported_dtype,
    _short_dtype,
    _short_ptype,
    _supported_ptypes,
    _supported_dtypes,
    _unsupported_ptype_map,
    _unsupported_dtype_map,
    _npy_to_itk_map,
    _itk_to_npy_map,
)

_supported_pclasses = {"scalar", "vector", "rgb", "rgba", "symmetric_second_rank_tensor"}
_image_type_map = {"scalar": "", "vector": "V", "rgb": "RGB", "rgba": "RGBA", "symmetric_second_rank_tensor": "SSRT" }

_image_read_dict = {}
for itype in {"scalar", "vector", "rgb", "rgba", "symmetric_second_rank_tensor"}:
    _image_read_dict[itype] = {}
    for p in _supported_ptypes:
        _image_read_dict[itype][p] = {}
        for d in {2, 3, 4}:
            ita = _image_type_map[itype]
            pa = _short_ptype(p)
            _image_read_dict[itype][p][d] = "imageRead%s%s%i" % (ita, pa, d)


def from_numpy(data, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False, pixeltype=None):
    """
    Create an ANTsImage object from a numpy array.

    The default pixel type is 'None', in which case the pixel type will be inferred from the numpy
    data pixel type. In general, the pixel type will be retained if supported. One exception to this
    is float64, which will be cast to float32 unless pixeltype is set to 'double'.

    With ``pixeltype=None`` and ``is_rgb=False``, the conversions are:

    ===================  ====================  ==================  ========================
    Input NumPy dtype    Output ANTs pixeltype  Output NumPy dtype  Precision
    ===================  ====================  ==================  ========================
    bool                 unsigned char         uint8               Exact
    uint8                unsigned char         uint8               Exact
    int8                 float                 float32             Exact
    int16                float                 float32             Exact
    uint16               unsigned int          uint32              Exact
    int32                float                 float32             May lose precision
    uint32               unsigned int          uint32              Exact
    int64                float                 float32             May lose precision
    float16              float                 float32             Exact
    float32              float                 float32             Unchanged
    float64              float                 float32             May lose range/precision
    Other (incl. uint64)  ValueError            --                  --
    ===================  ====================  ==================  ========================

    ANTsPy mostly uses float32 for computations. Float64 is not widely supported, but is allowed
    for :func:`ants.registration` and :func:`ants.apply_transforms`. For other operations, users
    should scale their data so that it can be represented as float32, before calling this function.

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
        whether the image has vector components.

    is_rgb : boolean
        whether the data is RGB. This implies has_components=True. Returned images always have type 'unsigned char'.

    pixeltype : str
        A supported pixel type for the returned image. If None, the pixel type will be inferred from the data.
        Where practical, the cast will be performed without loss of precision, however float64 will be cast
        to float32 by default. Use pixeltype=double to preserve double precision.

        If specified, the data will be cast to the requested type, which may result in a loss of precision.
        Options:
            'unsigned char' or 'uint8',
            'unsigned int' or 'uint32',
            'float' or 'float32',
            'double' or 'float64'

    Returns
    -------
    ANTsImage
        image with given data and any given metadata information.

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> data = np.random.rand(10, 10)
    >>> img = from_numpy(data) # float32
    >>> img = ants.image_read(ants.get_ants_data("r16")) # float32
    >>> img_arr = img.numpy() # float32
    >>> img2 = ants.from_numpy(img_arr, pixeltype="uint8")
    """
    if pixeltype is not None:
        requested_numpy_pixel_type = pixeltype
        if pixeltype in _itk_to_npy_map:
            requested_numpy_pixel_type = _itk_to_npy_map[pixeltype]
        if requested_numpy_pixel_type not in _supported_dtypes:
            raise ValueError(f"Unsupported pixel type: {pixeltype}")
        data = data.astype(requested_numpy_pixel_type)
    else:
        best_dtype = _get_best_supported_dtype(data.dtype)
        if best_dtype is None:
            raise ValueError(
                f"Cannot cast numpy type {data.dtype} to a supported ANTsImage pixel type. "
                f"Convert the data array to a supported numpy type before calling this function. "
                f"Supported types: {_supported_dtypes}"
            )
        data = data.astype(best_dtype, copy=False)

    # Be explicit about ordering of data - needs to be C-contiguous
    img = _from_numpy(data.T.copy(order='C'), origin, spacing, direction, has_components, is_rgb)
    return img


def _from_numpy(
    data, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False
):
    """
    Internal function for creating an ANTsImage. Note that data must be C-contiguous.
    The returned image may share data's buffer; callers must not modify data after passing
    it to this function.
    """
    if is_rgb:
        has_components = True
    ndim = data.ndim
    if has_components:
        ndim -= 1
    dtype = data.dtype.name

    data = np.asarray(data)

    if origin is None:
        origin = tuple([0.0] * ndim)
    if spacing is None:
        spacing = tuple([1.0] * ndim)
    if direction is None:
        direction = np.eye(ndim)

    libfn = get_lib_fn("fromNumpy%s%i" % (_short_dtype(dtype), ndim))

    if not has_components:
        itk_image = libfn(data, data.shape[::-1])
        ants_image = ants.from_pointer(itk_image)
        ants_image.set_origin(origin)
        ants_image.set_spacing(spacing)
        ants_image.set_direction(direction)
        ants_image._ndarr = data
    else:
        arrays = [data[i, ...].copy() for i in range(data.shape[0])]
        data_shape = arrays[0].shape
        ants_images = []
        for i in range(len(arrays)):
            tmp_ptr = libfn(arrays[i], data_shape[::-1])
            tmp_img = ants.from_pointer(tmp_ptr)
            tmp_img.set_origin(origin)
            tmp_img.set_spacing(spacing)
            tmp_img.set_direction(direction)
            tmp_img._ndarr = arrays[i]
            ants_images.append(tmp_img)
        ants_image = ants.merge_channels(ants_images)
        if is_rgb:
            ants_image = ants_image.vector_to_rgb()
    return ants_image


def make_image(
    shape,
    voxval=0,
    spacing=None,
    origin=None,
    direction=None,
    has_components=False,
    pixeltype="float",
):
    """
    Make an image with given size and voxel values or fill a mask with a vector of voxel values.

    ANTsR function: `makeImage`

    Arguments
    ---------
    shape : tuple/ANTsImage
        input image shape (tuple) or a mask (ANTsImage). If a mask, the output image has the same shape as the mask, and the
        number of voxel values in ``voxval`` should match the number of positive voxels in the mask. If a tuple, the number of
        voxel values in ``voxval`` should match the product of the dimensions in the tuple.

    voxval : scalar
        input image value. Either a scalar (single value) to be placed in all voxels, or a vector of with length matching the
        number of voxels required by the shape argument.

    spacing : tuple/list
        image spatial resolution.

    origin  : tuple/list
        image spatial origin.

    direction : list/ndarray
        direction matrix to convert from index to physical space.

    components : boolean
        whether there are components per pixel or not.

    pixeltype : str
        a supported pixel type for ANTsImage.

    Returns
    -------
    ANTsImage
    """
    if ants.is_image(shape):
        img = shape.clone()
        sel = shape > 0
        if voxval.ndim > 1:
            voxval = voxval.flatten()
        if (len(voxval) == int((sel > 0).sum())) or (len(voxval) == 0):
            img[sel] = voxval
        else:
            raise ValueError(
                "Num given voxels %i not same as num positive values %i in `shape`"
                % (len(voxval), int((sel > 0).sum()))
            )
        return img
    else:
        if isinstance(voxval, (tuple, list, np.ndarray)):
            array = np.asarray(voxval).astype("float32").reshape(shape)
        else:
            array = np.full(shape, voxval, dtype="float32")
        image = from_numpy(
            array,
            origin=origin,
            spacing=spacing,
            direction=direction,
            has_components=has_components,
        )
        return image.clone(pixeltype)


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
        raise Exception("filename does not exist")

    libfn = get_lib_fn("antsImageHeaderInfo")
    retval = libfn(filename)
    retval["dimensions"] = tuple(retval["dimensions"])
    retval["origin"] = tuple([round(o, 4) for o in retval["origin"]])
    retval["spacing"] = tuple([round(s, 4) for s in retval["spacing"]])
    retval["direction"] = np.round(retval["direction"], 4)
    return retval

def image_clone(image, pixeltype=None):
    """
    Clone an ANTsImage

    ANTsR function: `antsImageClone`

    Arguments
    ---------
    image : ANTsImage
        image to clone

    pixeltype : string (optional)
        new datatype for image

    Returns
    -------
    ANTsImage
    """
    return image.clone(pixeltype)


def image_read(filename, dimension=None, pixeltype="float", reorient=False):
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
        C++ or numpy datatype to be used to represent the pixels read. This datatype
        need not be the same as the datatype used in the file, but switching types may
        result in loss of precision or truncation. Options:
            'unsigned char' or 'uint8',
            'unsigned int' or 'uint32',
            'float' or 'float32',
            'double' or 'float64'

        Use pixeltype=None to set the pixeltype automatically. The returned
        image will use the datatype in the file if supported, otherwise it will
        attempt to convert an unsupported datatype to a supported type. An
        error is raised if the unsupported datatype cannot be safely converted.

        A pixeltype other than None will be used without checking whether it
        can precisely represent the data.

    reorient : boolean | string
        if True, the image will be reoriented to RPI if it is 3D
        if False, nothing will happen
        if string, this should be the 3-letter orientation to which the
            input image will reoriented if 3D.
        if the image is 2D, this argument is ignored

    Returns
    -------
    ANTsImage
    """
    if pixeltype is not None:
        # check pixeltype is supported and convert to itk/c++ name if needed
        if pixeltype not in _supported_ptypes and pixeltype not in _supported_dtypes:
            raise ValueError("Pixel type %s not supported!" % pixeltype)

        if pixeltype not in _supported_ptypes:
            pixeltype = _npy_to_itk_map.get(pixeltype)

    if filename.endswith(".npy"):
        filename = os.path.expanduser(filename)
        img_array = np.load(filename)
        if pixeltype is not None:
            img_array = img_array.astype(_itk_to_npy_map.get(pixeltype))
        if os.path.exists(filename.replace(".npy", ".json")):
            with open(filename.replace(".npy", ".json")) as json_data:
                img_header = json.load(json_data)
            ants_image = from_numpy(
                img_array,
                origin=img_header.get("origin", None),
                spacing=img_header.get("spacing", None),
                direction=np.asarray(img_header.get("direction", None)),
                has_components=img_header.get("components", 1) > 1,
                pixeltype=pixeltype
            )
        else:
            img_header = {}
            ants_image = from_numpy(img_array, pixeltype=pixeltype) # Still does type conversion if pixeltype=None

    else:
        filename = os.path.expanduser(filename)
        if not os.path.exists(filename):
            raise ValueError("File %s does not exist!" % filename)

        hinfo = image_header_info(filename)
        ptype = hinfo["pixeltype"]
        pclass = hinfo["pixelclass"]
        ndim = hinfo["nDimensions"]
        ncomp = hinfo["nComponents"]
        is_rgb = False
        if pclass == "rgb":
            pclass = "vector"
        if pclass == "rgba":
            pclass = "vector"
        if pclass == "symmetric_second_rank_tensor":
            pclass = "vector"
#        is_rgb = True if pclass == "rgb" else False
        if dimension is not None:
            ndim = dimension

        # error handling on pixelclass
        if pclass not in _supported_pclasses:
            raise ValueError("Pixel class %s not supported!" % pclass)

        # May need a different reader ptype to the ptype in the file
        reader_ptype = ptype
        if pixeltype is not None:
            # Read as requested type
            reader_ptype = pixeltype
        else:
            if ptype not in _supported_ptypes:
                reader_ptype = _unsupported_ptype_map.get(ptype)
            if reader_ptype is None:
                raise ValueError(
                    f"Cannot safely convert file pixel type {ptype} to a supported type. "
                    "You may force conversion by choosing a supported pixeltype, one of: "
                    f"{_supported_ptypes}"
                )

        # error handling on dimension
        if (ndim < 2) or (ndim > 4):
            raise ValueError("Found %i-dimensional image - not supported!" % ndim)

        libfn = get_lib_fn(_image_read_dict[pclass][reader_ptype][ndim])
        itk_pointer = libfn(filename)

        ants_image = ants.from_pointer(itk_pointer)

    if (reorient != False) and (ants_image.dimension == 3):
        if reorient == True:
            ants_image = ants_image.reorient_image2("RPI")
        elif isinstance(reorient, str):
            ants_image = ants_image.reorient_image2(reorient)

    return ants_image


def dicom_read(directory, pixeltype="float"):
    """
    Read a set of dicom files in a directory into a single ANTsImage.
    The origin of the resulting 3D image will be the origin of the
    first dicom image read.

    Arguments
    ---------
    directory : string
        folder in which all the dicom images exist

    pixeltype : string
        The pixel type to which the images should be converted. This is done at the
        file reader level and does not check that the requested type can represent
        the data.

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> img = ants.dicom_read('~/desktop/dicom-subject/')
    """
    slices = []
    imgidx = 0
    for imgpath in os.listdir(directory):
        if imgpath.endswith(".dcm"):
            if imgidx == 0:
                tmp = image_read(
                    os.path.join(directory, imgpath), dimension=3, pixeltype=pixeltype
                )
                origin = tmp.origin
                spacing = tmp.spacing
                direction = tmp.direction
                tmp = tmp.numpy()[:, :, 0]
            else:
                tmp = image_read(
                    os.path.join(directory, imgpath), dimension=2, pixeltype=pixeltype
                ).numpy()

            slices.append(tmp)
            imgidx += 1

    slices = np.stack(slices, axis=-1)
    return from_numpy(slices, origin=origin, spacing=spacing, direction=direction)

@image_method
def image_write(image, filename, ri=False):
    """
    Write an ANTsImage to file

    ANTsR function: `antsImageWrite`

    Arguments
    ---------
    image : ANTsImage
        image to save to file

    filename : string
        name of file to which image will be saved

    ri : boolean
        if True, return image. This allows for using this function in a pipeline:
            >>> img2 = img.smooth_image(2.).image_write(file1, ri=True).threshold_image(0,20).image_write(file2, ri=True)
        if False, do not return image
    """
    if filename.endswith(".npy"):
        img_array = image.numpy()
        img_header = {
            "origin": image.origin,
            "spacing": image.spacing,
            "direction": image.direction.tolist(),
            "components": image.components,
        }

        np.save(filename, img_array)
        with open(filename.replace(".npy", ".json"), "w") as outfile:
            json.dump(img_header, outfile)
    else:
        image.to_file(filename)

    if ri:
        return image

@image_method
def clone(image, pixeltype=None):
    """
    Create a copy of the given ANTsImage with the same data and info, possibly with
    a different data type for the image data. Only supports casting to
    uint8 (unsigned char), uint32 (unsigned int), float32 (float), and float64 (double)

    Arguments
    ---------
    pixeltype: string (optional)
        if None, the pixeltype will be the same as the cloned ANTsImage. Otherwise,
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
    if pixeltype is None:
        pixeltype = image.pixeltype

    if pixeltype not in _supported_ptypes:
        # check if the pixeltype is a numpy type
        if pixeltype in _supported_dtypes:
            pixeltype =  _npy_to_itk_map[pixeltype]
        else:
            raise ValueError('Pixeltype %s not supported. Supported types are %s' % (pixeltype, _supported_ptypes))

    if image.has_components and (not image.is_rgb):
        comp_imgs = ants.split_channels(image)
        comp_imgs_cloned = [comp_img.clone(pixeltype) for comp_img in comp_imgs]
        return ants.merge_channels(comp_imgs_cloned, channels_first=image.channels_first)
    else:
        p1_short = _short_ptype(image.pixeltype)
        p2_short = _short_ptype(pixeltype)
        ndim = image.dimension
        fn_suffix = '%s%i' % (p2_short,ndim)
        libfn = get_lib_fn('antsImageClone%s'%fn_suffix)
        pointer_cloned = libfn(image.pointer)
        return ants.from_pointer(pointer_cloned)

copy = clone

@image_method
def new_image_like(image, data):
    """
    Create a new ANTsImage with the same header information, but with
    a new image array.

    Arguments
    ---------
    data : ndarray or py::capsule
        New array or pointer for the image.
        It must have the same shape as the current
        image data.

    Returns
    -------
    ANTsImage
    """
    if not isinstance(data, np.ndarray):
        raise ValueError('data must be a numpy array')
    if not image.has_components:
        if data.shape != image.shape:
            raise ValueError('given array shape (%s) and image array shape (%s) do not match' % (data.shape, image.shape))
    else:
        if (data.shape[-1] != image.components) or (data.shape[:-1] != image.shape):
            raise ValueError('given array shape (%s) and image array shape (%s) do not match' % (data.shape[1:], image.shape))

    return from_numpy(data, origin=image.origin,
        spacing=image.spacing, direction=image.direction,
        has_components=image.has_components)

def from_numpy_like(data, image):
    return new_image_like(image, data)

def read_image_metadata(filename):
    """
    Read metadata dictionary from image file. This uses ITK ImageIO to efficiently read the metadata without reading the pixel
    data.

    Arguments
    ---------
    filename : string
        name of image file from which metadata will be read.

    Returns
    -------
    dict
    """
    if not os.path.exists(filename):
        raise Exception("filename does not exist")

    libfn = get_lib_fn("readImageMetadata")
    retval = libfn(filename)
    return retval
