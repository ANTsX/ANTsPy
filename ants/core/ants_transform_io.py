
__all__ = ['create_ants_transform',
           'new_ants_transform',
           'read_transform',
           'write_transform',
           'transform_from_displacement_field']

import os
import numpy as np

from . import ants_image as iio
from . import ants_transform as tio
from .. import lib

_new_ants_transform_dict = {}
for ttype,tabbrev in zip(['float','double'],['F','D']):
    _new_ants_transform_dict[ttype] = {}
    for d in {2,3,4}:
        _new_ants_transform_dict[ttype][d] = 'new_ants_transform%s%i' % (tabbrev, d)

_read_transform_dict = {}
for ttype,tabbrev in zip(['float','double'],['F','D']):
    _read_transform_dict[ttype] = {}
    for d in {2,3,4}:
        _read_transform_dict[ttype][d] = 'readTransform%s%i' % (tabbrev, d)

_write_transform_dict = {}
for ttype,tabbrev in zip(['float','double'],['F','D']):
    _write_transform_dict[ttype] = {}
    for d in {2,3,4}:
        _write_transform_dict[ttype][d] = 'writeTransform%s%i' % (tabbrev, d)

_matrix_offset_dict = {}
for ttype,tabbrev in zip(['float','double'],['F','D']):
    _matrix_offset_dict[ttype] = {}
    for d in {2,3,4}:
        _matrix_offset_dict[ttype][d] = 'matrixOffset%s%i' % (tabbrev, d)


def new_ants_transform(precision='float', dimension=3, transform_type='AffineTransform', parameters=None):
    """
    Create a new ANTsTransform

    ANTsR function: None
    """
    new_ants_transform_fn = lib.__dict__[_new_ants_transform_dict[precision][dimension]]

    itk_tx = new_ants_transform_fn(precision, dimension, transform_type)
    ants_tx = tio.ANTsTransform(itk_tx)

    if parameters is not None:
        ants_tx.set_parameters(parameters)

    return ants_tx


def create_ants_transform(transform_type='AffineTransform',
                          precision='float', 
                          dimension=3,
                          matrix=None,
                          offset=None, 
                          center=None, 
                          translation=None, 
                          parameters=None, 
                          fixed_parameters=None, 
                          displacement_field=None,
                          supported_types=False):
    """
    Create and initialize an ANTsTransform

    ANTsR function: `createAntsrTransform`

    Arguments
    ---------
    transform_type : string   
        type of transform(s)
    
    precision : string
        numerical precision
    
    dimension : integer
        spatial dimension of transform
    
    matrix : ndarray
        matrix for linear transforms
    
    offset : tuple/list
        offset for linear transforms
    
    center : tuple/list
        center for linear transforms
    
    translation : tuple/list
        translation for linear transforms
    
    parameters : ndarray/list
        array of parameters
    
    fixed_parameters : ndarray/list   
        array of fixed parameters
    
    displacement_field : ANTsImage
        multichannel ANTsImage for non-linear transform
    
    supported_types : boolean
        flag that returns array of possible transforms types

    Returns
    -------
    ANTsTransform or list of ANTsTransform types

    Example
    -------
    >>> import ants
    >>> translation = (3,4,5)
    >>> tx = ants.create_ants_transform( type='Euler3DTransform', translation=translation )
    """
    def _check_arg(arg, dim=1):
        if arg is None:
            if dim == 1:
                return []
            elif dim == 2:
                return [[]]
        elif isinstance(arg, np.ndarray):
            return arg.tolist()
        elif isinstance(arg, (tuple, list)):
            return list(arg)
        else:
            raise ValueError('Incompatible input argument')

    matrix = _check_arg(matrix, dim=2)
    offset = _check_arg(offset)
    center = _check_arg(center)
    translation = _check_arg(translation)
    parameters = _check_arg(parameters)
    fixed_parameters = _check_arg(fixed_parameters)

    matrix_offset_types = {'AffineTransform', 'CenteredAffineTransform', 
                         'Euler2DTransform', 'Euler3DTransform', 
                         'Rigid2DTransform', 'QuaternionRigidTransform', 
                         'Similarity2DTransform', 'CenteredSimilarity2DTransform',
                         'Similarity3DTransform', 'CenteredRigid2DTransform', 
                         'CenteredEuler3DTransform'}

    if supported_types:
      return set(list(matrix_offset_types) + ['DisplacementFieldTransform'])

    # Check for valid dimension
    if (dimension < 2) or (dimension > 4):
        raise ValueError('Unsupported dimension: %i' % dimension)

    # Check for valid precision
    precision_types = ('float', 'double')
    if precision not in precision_types:
        raise ValueError('Unsupported Precision %s' % str(precision))

    # Check for supported transform type
    if (transform_type not in matrix_offset_types) and (transform_type != 'DisplacementFieldTransform'):
        raise ValueError('Unsupported type %s' % str(transform_type)) 

    # Check parameters with type
    if (transform_type=='Euler3DTransform'):
        dimension = 3
    elif (transform_type=='Euler2DTransform'):
        dimension = 2
    elif (transform_type=='Rigid3DTransform'):
        dimension = 3
    elif (transform_type=='QuaternionRigidTransform'):
        dimension = 3
    elif (transform_type=='Rigid2DTransform'):
        dimension = 2
    elif (transform_type=='CenteredRigid2DTransform'):
        dimension = 2
    elif (transform_type=='CenteredEuler3DTransform'):
        dimension = 3
    elif (transform_type=='Similarity3DTransform'):
        dimension = 3
    elif (transform_type=='Similarity2DTransform'):
        dimension = 2
    elif (transform_type=='CenteredSimilarity2DTransform'):
        dimension = 2

    # If displacement field
    if displacement_field is not None:
        raise ValueError('Displacement field transform not currently supported')
    #    itk_tx = transform_from_displacement_field(displacement_field)
    #    return tio.ants_transform(itk_tx)

    # Transforms that derive from itk::MatrixOffsetTransformBase
    elif transform_type in matrix_offset_types:
        matrix_offset_fn = lib.__dict__[_matrix_offset_dict[precision][dimension]]
        itk_tx = matrix_offset_fn(transform_type,
                                  precision,
                                  dimension,
                                  matrix,
                                  offset,
                                  center,
                                  translation,
                                  parameters,
                                  fixed_parameters)
        return tio.ANTsTransform(itk_tx)
    else:
        raise ValueError('transform_type not supported or unkown error happened')


def transform_from_displacement_field(field):
    """
    Convert deformation field (multiChannel image) to ANTsTransform

    ANTsR function: `antsrTransformFromDisplacementField`

    Arguments
    ---------
    field : ANTsImage
        deformation field as multi-channel ANTsImage

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16') )
    >>> mi = ants.image_read(ants.get_ants_data('r64') )
    >>> fi = ants.resample_image(fi,(60,60),1,0)
    >>> mi = ants.resample_image(mi,(60,60),1,0) # speed up
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
    >>> compfield = ants.compose_transforms_to_field( fi, mytx['fwd'] )
    >>> atx = ants.transform_from_displacement_field( compfield )
    """
    if not isinstance(field, iio.ANTsImage):
        raise ValueError('field must be ANTsImage type')
    if field.dimension == 2:
        field = field.clone('float')
        return tio.ANTsTransform(lib.antsTransformFromDisplacementFieldF2(field.pointer))
    elif field.dimension == 3:
        field = field.clone('float')
        return tio.ANTsTransform(lib.antsrTransformFromDisplacementFieldF3(field.pointer))


def read_transform(filename, dimension=3, precision='float'):
    """
    Read a transform from file

    ANTsR function: `readAntsrTransform`

    Arguments
    ---------
    filename : string
        filename of transform

    dimension : integer
        spatial dimension of transform

    precision : string
        numerical precision of transform
    
    Returns
    -------
    ANTsTransform

    Example
    -------
    >>> import ants
    >>> tx = ants.new_ants_transform(dimension=2)
    >>> tx.set_parameters((0.9,0,0,1.1,10,11))
    >>> ants.write_transform(tx, '~/desktop/tx.mat')
    >>> tx2 = ants.read_transform('~/desktop/tx.mat')
    """
    filename = os.path.expanduser(filename)
    read_transform_fn = lib.__dict__[_read_transform_dict[precision][dimension]]
    itk_tx = read_transform_fn(filename, dimension, precision)
    return tio.ANTsTransform(itk_tx)


def write_transform(transform, filename):
    """
    Write ANTsTransform to file

    ANTsR function: `writeAntsrTransform`

    Arguments
    ---------
    transform : ANTsTransform
        transform to save

    filename : string
        filename of transform (file extension is ".mat" for affine transforms)
    
    Returns
    -------
    N/A

    Example
    -------
    >>> import ants
    >>> tx = ants.new_ants_transform(dimension=2)
    >>> tx.set_parameters((0.9,0,0,1.1,10,11))
    >>> ants.write_transform(tx, '~/desktop/tx.mat')
    >>> tx2 = ants.read_transform('~/desktop/tx.mat')
    """
    filename = os.path.expanduser(filename)
    write_transform_fn = lib.__dict__[_write_transform_dict[transform.precision][transform.dimension]]
    write_transform_fn(transform._tx, filename)


