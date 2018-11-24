

import numpy as np

__all__ = ['ANTsTransform',
           'set_ants_transform_parameters',
           'get_ants_transform_parameters',
           'get_ants_transform_fixed_parameters',
           'set_ants_transform_fixed_parameters',
           'apply_ants_transform',
           'apply_ants_transform_to_point',
           'apply_ants_transform_to_vector',
           'apply_ants_transform_to_image',
           'invert_ants_transform',
           'compose_ants_transforms',
           'transform_index_to_physical_point',
           'transform_physical_point_to_index']

from . import ants_image as iio
from .. import utils


class ANTsTransform(object):

    def __init__(self, precision='float', dimension=3, transform_type='AffineTransform', pointer=None):
        """
        NOTE: This class should never be initialized directly by the user.

        Initialize an ANTsTransform object.

        Arguments
        ---------
        pixeltype : string

        dimension : integer

        transform_type : stirng

        pointer : py::capsule (optional)
        """
        self.precision = precision
        self.dimension = dimension
        self.transform_type = transform_type
        self.type = transform_type

        if pointer is None:
            libfn = utils.get_lib_fn('newAntsTransform%s%i' % (utils.short_ptype(precision), dimension))
            pointer = libfn(precision, dimension, transform_type)

        self.pointer = pointer

        # suffix to append to all c++ library function calls
        self._libsuffix = '%s%i' % (utils.short_ptype(self.precision), self.dimension)

    @property
    def parameters(self):
        """ Get parameters of transform """
        libfn = utils.get_lib_fn('getTransformParameters%s'%self._libsuffix)
        return np.asarray(libfn(self.pointer), order='F')#.reshape((self.dimension, self.dimension+1), order='F')

    def set_parameters(self, parameters):
        """ Set parameters of transform """
        if not isinstance(parameters, np.ndarray):
            parameters = np.asarray(parameters)

        # if in two dimensions, flatten with fortran ordering
        if parameters.ndim > 1:
            parameters = parameters.flatten(order='F')

        libfn = utils.get_lib_fn('setTransformParameters%s'%self._libsuffix)
        libfn(self.pointer, parameters.tolist())

    @property
    def fixed_parameters(self):
        """ Get parameters of transform """
        libfn = utils.get_lib_fn('getTransformFixedParameters%s'%self._libsuffix)
        return np.asarray(libfn(self.pointer))

    def set_fixed_parameters(self, parameters):
        """ Set parameters of transform """
        if not isinstance(parameters, np.ndarray):
            parameters = np.asarray(parameters)

        libfn = utils.get_lib_fn('setTransformFixedParameters%s'%self._libsuffix)
        libfn(self.pointer, parameters.tolist())

    def invert(self):
        """ Invert the transform """
        libfn = utils.get_lib_fn('inverseTransform%s' % (self._libsuffix))
        inv_tx_ptr = libfn(self.pointer)

        new_tx = ANTsTransform(precision=self.precision, dimension=self.dimension,
                                transform_type=self.transform_type, pointer=inv_tx_ptr)

        return new_tx

    def apply(self, data, data_type='point', reference=None, **kwargs):
        """
        Apply transform to data

        """
        if data_type == 'point':
            return self.apply_to_point(data)
        elif data_type == 'vector':
            return self.apply_to_vector(data)
        elif data_type == 'image':
            return self.apply_to_image(data, reference, **kwargs)

    def apply_to_point(self, point):
        """
        Apply transform to a point

        Arguments
        ---------
        point : list/tuple
            point to which the transform will be applied

        Returns
        -------
        list : transformed point

        Example
        -------
        >>> import ants
        >>> tx = ants.new_ants_transform()
        >>> params = tx.parameters
        >>> tx.set_parameters(params*2)
        >>> pt2 = tx.apply_to_point((1,2,3)) # should be (2,4,6)
        """
        libfn = utils.get_lib_fn('transformPoint%s' % (self._libsuffix))
        return tuple(libfn(self.pointer, point))

    def apply_to_vector(self, vector):
        """
        Apply transform to a vector

        Arguments
        ---------
        vector : list/tuple
            vector to which the transform will be applied

        Returns
        -------
        list : transformed vector
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        libfn = utils.get_lib_fn('transformVector%s' % (self._libsuffix))
        return np.asarray(libfn(self.pointer, vector))

    def apply_to_image(self, image, reference=None, interpolation='linear'):
        """
        Apply transform to an image

        Arguments
        ---------
        image : ANTsImage
            image to which the transform will be applied

        reference : ANTsImage
            target space for transforming image

        interpolation : string
            type of interpolation to use

        Returns
        -------
        list : transformed vector

        """
        if reference is None:
            reference = image.clone()

        tform_fn = utils.get_lib_fn('transformImage%s%s' % (self._libsuffix, image._libsuffix))
        reference = reference.clone(image.pixeltype)

        img_ptr = tform_fn(self.pointer, image.pointer, reference.pointer, interpolation)
        return iio.ANTsImage(pixeltype=image.pixeltype,
                            dimension=image.dimension,
                            components=image.components,
                            pointer=img_ptr)

    def __repr__(self):
        s = "ANTsTransform\n" +\
            '\t {:<10} : {}\n'.format('Type', self.type)+\
            '\t {:<10} : {}\n'.format('Dimension', self.dimension)+\
            '\t {:<10} : {}\n'.format('Precision', self.precision)
        return s

# verbose functions for ANTsR compatibility
def set_ants_transform_parameters(transform, parameters):
    """
    Set parameters of an ANTsTransform

    ANTsR function: `setAntsrTransformParameters`
    """
    transform.set_parameters(parameters)

def get_ants_transform_parameters(transform):
    """
    Get parameters of an ANTsTransform

    ANTsR function: `getAntsrTransformParameters`
    """
    return transform.parameters

def set_ants_transform_fixed_parameters(transform, parameters):
    """
    Set fixed parameters of an ANTsTransform

    ANTsR function: `setAntsrTransformFixedParameters`
    """
    transform.set_fixed_parameters(parameters)

def get_ants_transform_fixed_parameters(transform):
    """
    Get fixed parameters of an ANTsTransform

    ANTsR function: `getAntsrTransformFixedParameters`
    """
    return transform.fixed_parameters


def apply_ants_transform(transform, data, data_type="point", reference=None, **kwargs):
    """
    Apply ANTsTransform to data

    ANTsR function: `applyAntsrTransform`

    Arguments
    ---------
    transform : ANTsTransform
        transform to apply to image

    data : ndarray/list/tuple
        data to which transform will be applied

    data_type : string
        type of data
        Options :
            'point'
            'vector'
            'image'

    reference : ANTsImage
        target space for transforming image

    kwargs : kwargs
        additional options passed to `apply_ants_transform_to_image`

    Returns
    -------
    ANTsImage if data_type == 'point'
    OR
    tuple if data_type == 'point' or data_type == 'vector'
    """
    return transform.apply(data, data_type, reference, **kwargs)


def apply_ants_transform_to_point(transform, point):
    """
    Apply transform to a point

    ANTsR function: `applyAntsrTransformToPoint`

    Arguments
    ---------
    point : list/tuple
        point to which the transform will be applied

    Returns
    -------
    tuple : transformed point

    Example
    -------
    >>> import ants
    >>> tx = ants.new_ants_transform()
    >>> params = tx.parameters
    >>> tx.set_parameters(params*2)
    >>> pt2 = tx.apply_to_point((1,2,3)) # should be (2,4,6)
    """
    return transform.apply_to_point(point)


def apply_ants_transform_to_vector(transform, vector):
    """
    Apply transform to a vector

    ANTsR function: `applyAntsrTransformToVector`

    Arguments
    ---------
    vector : list/tuple
        vector to which the transform will be applied

    Returns
    -------
    tuple : transformed vector
    """
    return transform.apply_to_vector(vector)


def apply_ants_transform_to_image(transform, image, reference, interpolation='linear'):
    """
    Apply transform to an image

    ANTsR function: `applyAntsrTransformToImage`

    Arguments
    ---------
    image : ANTsImage
        image to which the transform will be applied

    reference : ANTsImage
        reference image

    interpolation : string
        type of interpolation to use

    Returns
    -------
    list : transformed vector

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data("r16")).clone('float')
    >>> tx = ants.new_ants_transform(dimension=2)
    >>> tx.set_parameters((0.9,0,0,1.1,10,11))
    >>> img2 = tx.apply_to_image(img, img)
    """
    return transform.apply_to_image(image, reference, interpolation)


def invert_ants_transform(transform):
    """
    Invert ANTsTransform

    ANTsR function: `invertAntsrTransform`

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data("r16")).clone('float')
    >>> tx = ants.new_ants_transform(dimension=2)
    >>> tx.set_parameters((0.9,0,0,1.1,10,11))
    >>> img_transformed = tx.apply_to_image(img, img)
    >>> inv_tx = tx.invert()
    >>> img_orig = inv_tx.apply_to_image(img_transformed, img_transformed)
    """
    return transform.invert()

def compose_ants_transforms(transform_list):
    """
    Compose multiple ANTsTransform's together

    ANTsR function: `composeAntsrTransforms`

    Arguments
    ---------
    transform_list : list/tuple of ANTsTransform object
        list of transforms to compose together

    Returns
    -------
    ANTsTransform
        one transform that contains all given transforms

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data("r16")).clone('float')
    >>> tx = ants.new_ants_transform(dimension=2)
    >>> tx.set_parameters((0.9,0,0,1.1,10,11))
    >>> inv_tx = tx.invert()
    >>> single_tx = ants.compose_ants_transforms([tx, inv_tx])
    >>> img_orig = single_tx.apply_to_image(img, img)
    >>> rRotGenerator = ants.contrib.RandomRotate2D( ( 0, 40 ), reference=img )
    >>> rShearGenerator=ants.contrib.RandomShear2D( (0,50), reference=img )
    >>> tx1 = rRotGenerator.transform()
    >>> tx2 = rShearGenerator.transform()
    >>> rSrR = ants.compose_ants_transforms([tx1, tx2])
    >>> rSrR.apply_to_image( img )
    """
    precision = transform_list[0].precision
    dimension = transform_list[0].dimension

    for tx in transform_list:
        if precision != tx.precision:
            raise ValueError('All transforms must have the same precision')
        if dimension != tx.dimension:
            raise ValueError('All transforms must have the same dimension')

    tx_ptr_list = list(reversed([tf.pointer for tf in transform_list]))
    libfn = utils.get_lib_fn('composeTransforms%s' % (transform_list[0]._libsuffix))
    itk_composed_tx = libfn(tx_ptr_list, precision, dimension)
    return ANTsTransform(precision=precision, dimension=dimension,
                        transform_type='CompositeTransform', pointer=itk_composed_tx)


def transform_index_to_physical_point(image, index):
    """
    Get spatial point from index of an image.

    ANTsR function: `antsTransformIndexToPhysicalPoint`

    Arguments
    ---------
    img : ANTsImage
        image to get values from

    index : list or tuple or numpy.ndarray
        location in image

    Returns
    -------
    tuple

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> img = ants.make_image((10,10),np.random.randn(100))
    >>> pt = ants.transform_index_to_physical_point(img, (2,2))
    """
    if not isinstance(image, iio.ANTsImage):
        raise ValueError('image must be ANTsImage type')

    if isinstance(index, np.ndarray):
        index = index.tolist()

    if not isinstance(index, (tuple,list)):
        raise ValueError('index must be tuple or list')

    if len(index) != image.dimension:
        raise ValueError('len(index) != image.dimension')

    index = [i+1 for i in index]
    ndim = image.dimension
    ptype = image.pixeltype
    libfn = utils.get_lib_fn('TransformIndexToPhysicalPoint%s%i' % (utils.short_ptype(ptype), ndim))
    point = libfn(image.pointer, [list(index)])

    return np.array(point[0])


def transform_physical_point_to_index(image, point):
    """
    Get index from spatial point of an image.

    ANTsR function: `antsTransformPhysicalPointToIndex`

    Arguments
    ---------
    image : ANTsImage
        image to get values from

    point : list or tuple or numpy.ndarray
        point in image

    Returns
    -------
    tuple

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> img = ants.make_image((10,10),np.random.randn(100))
    >>> idx = ants.transform_physical_point_to_index(img, (2,2))
    >>> img.set_spacing((2,2))
    >>> idx2 = ants.transform_physical_point_to_index(img, (4,4))
    """
    if not isinstance(image, iio.ANTsImage):
        raise ValueError('image must be ANTsImage type')

    if isinstance(point, np.ndarray):
        point = point.tolist()

    if not isinstance(point, (list,tuple)):
        raise ValueError('point must be list, tuple, or np.ndarray')

    if len(point) != image.dimension:
        raise ValueError('len(index) != image.dimension')

    ndim = image.dimension
    ptype = image.pixeltype
    libfn = utils.get_lib_fn('TransformPhysicalPointToIndex%s%i'%(utils.short_ptype(ptype),ndim))
    index = libfn(image.pointer, [list(point)])
    index = [i-1 for i in index[0]]

    return np.array(index)
