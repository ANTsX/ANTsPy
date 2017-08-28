

import numpy as np

__all__ = ['set_ants_transform_parameters',
           'get_ants_transform_parameters',
           'get_ants_transform_fixed_parameters',
           'apply_ants_transform',
           'apply_ants_transform_to_point',
           'apply_ants_transform_to_vector',
           'apply_ants_transform_to_image',
           'invert_ants_transform',
           'compose_ants_transforms']

from . import ants_image
from .. import lib

_compose_transforms_dict = {
    'float': {
        2: lib.composeTransformsF2,
        3: lib.composeTransformsF3,
        4: lib.composeTransformsF4
    },
    'double': {
        2: lib.composeTransformsD2,
        3: lib.composeTransformsD3,
        4: lib.composeTransformsD4
    }
}

class ANTsTransform(object):

    def __init__(self, tx):
        """
        Initialize an ANTsTransform object
        """
        self._tx = tx
        self.precision = tx.precision
        self.dimension = tx.dimension
        self.type = tx.type
        self.pointer = tx.pointer

    @property
    def parameters(self):
        return np.asarray(self._tx.get_parameters())

    def set_parameters(self, parameters):
        if isinstance(parameters, np.ndarray):
            parameters = parameters.tolist()
        self._tx.set_parameters(parameters)

    @property
    def fixed_parameters(self):
        return np.asarray(self._tx.get_fixed_parameters())

    def set_fixed_parameters(self, parameters):
        if isinstance(parameters, np.ndarray):
            parameters = parameters.tolist()
        self._tx.set_fixed_parameters(parameters)

    def invert(self):
        return ANTsTransform(self._tx.inverse())

    def apply(self, data, data_type='point', reference=None, **kwargs):
        if data_type == 'point':
            return self.apply_to_point(data)
        elif data_type == 'vector':
            return self.apply_to_vector(data)
        elif data_type == 'image':
            return self.apply_to_image(data, reference, **kwargs)

    def apply_to_point(self, point):
        return tuple(self._tx.transform_point(point))

    def apply_to_vector(self, vector):
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return np.asarray(self._tx.transform_vector(vector))

    def apply_to_image(self, image, reference, interpolation='linear'):
        if image.pixeltype == 'unsigned char':
            tform_fn = self._tx.transform_imageUC
        elif image.pixeltype == 'char':
            tform_fn = self._tx.transform_imageC
        elif image.pixeltype == 'unsigned short':
            tform_fn = self._tx.transform_imageUS
        elif image.pixeltype == 'short':
            tform_fn = self._tx.transform_imageS
        elif image.pixeltype == 'unsigned int':
            tform_fn = self._tx.transform_imageUI
        elif image.pixeltype == 'int':
            tform_fn = self._tx.transform_imageI
        elif image.pixeltype == 'float':
            tform_fn = self._tx.transform_imageF
        elif image.pixeltype == 'double':
            tform_fn = self._tx.transform_imageD

        reference = reference.clone(image.pixeltype)

        return ants_image.ANTsImage(tform_fn(image._img, reference._img, interpolation))


# verbose functions for ANTsR compatibility
def set_ants_transform_parameters(transform, parameters):
    transform.set_parameters(parameters)

def get_ants_transform_parameters(transform):
    return transform.get_parameters()

def get_ants_transform_fixed_parameters(transform):
    return transform.get_fixed_parameters()

def set_ants_transform_fixed_parameters(transform, parameters):
    transform.set_fixed_parameters(parameters)


def apply_ants_transform(transform, data, data_type="point", reference=None, **kwargs):
    return transform.apply_transform(data, data_type, reference, **kwargs)


def apply_ants_transform_to_point(transform, point):
    """
    Example
    -------
    >>> import ants
    >>> tx = ants.new_ants_transform()
    >>> params = tx.parameters
    >>> tx.set_parameters(params*2)
    >>> pt2 = tx.apply_to_point((1,2,3)) # should be (2,4,6)
    """
    return transform.apply_transform_to_point(point)


def apply_ants_transform_to_vector(transform, vector):
    return transform.apply_to_vector(vector)


def apply_ants_transform_to_image(transform, image, reference, interpolation='linear'):
    """
    Apply an ANTsTransform to an ANTsImage

    Arguments
    ---------
    transform : ANTsTransform object

    Example
    -------
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data("r16")).clone('float')
    >>> tx = ants.new_ants_transform(dimension=2)
    >>> tx.set_parameters((0.9,0,0,1.1,10,11))
    >>> img2 = tx.apply_to_image(img, img)
    """
    return transform.apply_transform_to_image(image, reference, interpolation)


def invert_ants_transform(transform):
    """
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
    >>> import ants
    >>> img = ants.image_read(ants.get_ants_data("r16")).clone('float')
    >>> tx = ants.new_ants_transform(dimension=2)
    >>> tx.set_parameters((0.9,0,0,1.1,10,11))
    >>> inv_tx = tx.invert()
    >>> single_tx = ants.compose_ants_transforms([tx, inv_tx])
    >>> img_orig = single_tx.apply_to_image(img, img)
    """
    precision = transform_list[0].precision
    dimension = transform_list[0].dimension

    for tx in transform_list:
        if precision != tx.precision:
            raise ValueError('All transforms must have the same precision')
        if dimension != tx.dimension:
            raise ValueError('All transforms must have the same dimension')

    transform_list = list(reversed([tf._tx for tf in transform_list]))
    compose_transform_fn = _compose_transforms_dict[precision][dimension]

    itk_composed_tx = compose_transform_fn(transform_list, precision, dimension)
    return ANTsTransform(itk_composed_tx)



