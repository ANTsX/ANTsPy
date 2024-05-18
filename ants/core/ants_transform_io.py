__all__ = [
    "create_ants_transform",
    "new_ants_transform",
    "read_transform",
    "write_transform",
    "transform_from_displacement_field",
    "transform_to_displacement_field",
    "fsl2antstransform"
]

import os
import numpy as np

import ants
from ants.internal import get_lib_fn, short_ptype


def new_ants_transform(
    precision="float", dimension=3, transform_type="AffineTransform", parameters=None,
    fixed_parameters=None
):
    """
    Create a new ANTsTransform

    ANTsR function: None

    This is a simplified method for creating an ANTsTransform, mostly used internally.
    See create_ants_transform for more options.

    Example
    -------
    >>> import ants
    >>> tx = ants.new_ants_transform()
    """
    libfn = get_lib_fn(
        "newAntsTransform%s%i" % (short_ptype(precision), dimension)
    )
    itk_tx = libfn(precision, dimension, transform_type)
    ants_tx = ants.ANTsTransform(
        precision=precision,
        dimension=dimension,
        transform_type=transform_type,
        pointer=itk_tx,
    )

    if parameters is not None:
        ants_tx.set_parameters(parameters)

    if fixed_parameters is not None:
        ants_tx.set_fixed_parameters(fixed_parameters)

    return ants_tx


def create_ants_transform(
    transform_type="AffineTransform",
    precision="float",
    dimension=3,
    matrix=None,
    offset=None,
    center=None,
    translation=None,
    parameters=None,
    fixed_parameters=None,
    displacement_field=None,
    supported_types=False,
):
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
            raise ValueError("Incompatible input argument")

    matrix = _check_arg(matrix, dim=2)
    offset = _check_arg(offset)
    center = _check_arg(center)
    translation = _check_arg(translation)
    parameters = _check_arg(parameters)
    fixed_parameters = _check_arg(fixed_parameters)

    matrix_offset_types = {
        "AffineTransform",
        "CenteredAffineTransform",
        "Euler2DTransform",
        "Euler3DTransform",
        "Rigid3DTransform",
        "Rigid2DTransform",
        "QuaternionRigidTransform",
        "Similarity2DTransform",
        "CenteredSimilarity2DTransform",
        "Similarity3DTransform",
        "CenteredRigid2DTransform",
        "CenteredEuler3DTransform",
    }

    # user_matrix_types = {'Affine','CenteredAffine',
    #                     'Euler', 'CenteredEuler',
    #                     'Rigid', 'CenteredRigid', 'QuaternionRigid',
    #                     'Similarity', 'CenteredSimilarity'}

    if supported_types:
        return set(list(matrix_offset_types) + ["DisplacementFieldTransform"])

    # Check for valid dimension
    if (dimension < 2) or (dimension > 4):
        raise ValueError("Unsupported dimension: %i" % dimension)

    # Check for valid precision
    precision_types = ("float", "double")
    if precision not in precision_types:
        raise ValueError("Unsupported Precision %s" % str(precision))

    # Check for supported transform type
    if (transform_type not in matrix_offset_types) and (
        transform_type != "DisplacementFieldTransform"
    ):
        raise ValueError("Unsupported type %s" % str(transform_type))

    # Check parameters with type
    if transform_type == "Euler3DTransform":
        dimension = 3
    elif transform_type == "Euler2DTransform":
        dimension = 2
    elif transform_type == "Rigid3DTransform":
        dimension = 3
    elif transform_type == "QuaternionRigidTransform":
        dimension = 3
    elif transform_type == "Rigid2DTransform":
        dimension = 2
    elif transform_type == "CenteredRigid2DTransform":
        dimension = 2
    elif transform_type == "CenteredEuler3DTransform":
        dimension = 3
    elif transform_type == "Similarity3DTransform":
        dimension = 3
    elif transform_type == "Similarity2DTransform":
        dimension = 2
    elif transform_type == "CenteredSimilarity2DTransform":
        dimension = 2

    # If displacement field
    if displacement_field is not None:
        # raise ValueError('Displacement field transform not currently supported')
        itk_tx = transform_from_displacement_field(displacement_field)
        return ants.ants_transform(itk_tx)

    # Transforms that derive from itk::MatrixOffsetTransformBase
    libfn = get_lib_fn(
        "matrixOffset%s%i" % (short_ptype(precision), dimension)
    )
    itk_tx = libfn(
        transform_type,
        precision,
        dimension,
        matrix,
        offset,
        center,
        translation,
        parameters,
        fixed_parameters,
    )
    return ants.ANTsTransform(
        precision=precision,
        dimension=dimension,
        transform_type=transform_type,
        pointer=itk_tx,
    )


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
    >>> vec = ants.image_read( mytx['fwdtransforms'][0] )
    >>> atx = ants.transform_from_displacement_field( vec )
    """
    if not ants.is_image(field):
        raise ValueError("field must be ANTsImage type")
    if field.dimension < 2 or field.dimension > 3:
        raise ValueError("Unsupported displacement field dimension: %i" % field.dimension)
    if field.components != field.dimension:
        raise ValueError("Displacement field must have same number of components as the image dimension")
    libfn = get_lib_fn("antsTransformFromDisplacementField")
    field = field.clone("float")
    txptr = libfn(field.pointer)
    return ants.ANTsTransform(
        precision="float",
        dimension=field.dimension,
        transform_type="DisplacementFieldTransform",
        pointer=txptr,
    )


def transform_to_displacement_field(xfrm, ref):
    """
    Convert displacement field ANTsTransform to displacement field

    ANTsR function: `antsrTransformToDisplacementField`

    Arguments
    ---------
    xfrm : displacement field ANTsTransform
        displacement field ANTsTransform

    ref : ANTs Image

    Returns
    -------
    ANTsVectorImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16') )
    >>> mi = ants.image_read(ants.get_ants_data('r64') )
    >>> fi = ants.resample_image(fi,(60,60),1,0)
    >>> mi = ants.resample_image(mi,(60,60),1,0) # speed up
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
    >>> vec = ants.image_read( mytx['fwdtransforms'][0] )
    >>> atx = ants.transform_from_displacement_field( vec )
    >>> field = ants.transform_to_displacement_field( atx, fi )
    """

    if not xfrm.type == 'DisplacementFieldTransform':
        raise ValueError("Transform must be of DisplacementFieldTransform type")
    libfn = get_lib_fn("antsTransformToDisplacementField")
    field_ptr = libfn(xfrm.pointer, ref.pointer)
    return ants.from_pointer(field_ptr)


def read_transform(filename, precision="float"):
    """
    Read a transform from file

    ANTsR function: `readAntsrTransform`

    Arguments
    ---------
    filename : string
        filename of transform

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
    if not os.path.exists(filename):
        raise ValueError("filename does not exist!")

    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        return transform_from_displacement_field(ants.image_read(filename))

    libfn1 = get_lib_fn("getTransformDimensionFromFile")
    dimensionUse = libfn1(filename)

    libfn2 = get_lib_fn("getTransformNameFromFile")
    transform_type = libfn2(filename)

    libfn3 = get_lib_fn(
        "readTransform%s%i" % (short_ptype(precision), dimensionUse)
    )
    itk_tx = libfn3(filename, dimensionUse, precision)

    return ants.ANTsTransform(
        precision=precision,
        dimension=dimensionUse,
        transform_type=transform_type,
        pointer=itk_tx,
    )


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
    if not isinstance(transform, ants.ANTsTransform):
        raise Exception('Only ANTsTransform instances can be written to file. Check that you are not passing in a filepath to a saved transform.')
    filename = os.path.expanduser(filename)
    libfn = get_lib_fn("writeTransform")
    libfn(transform.pointer, filename)

def fsl2antstransform(matrix, reference, moving):
    """
    Convert an FSL linear transform to an antsrTransform
    
    ANTsR function: `fsl2antsrtransform`

    Arguments
    ---------
    matrix : ndarray/list
        4x4 matrix of transform parameters

    reference : ANTsImage
        target image

    moving : ANTsImage
        moving image

    Returns
    -------
    ANTsTransform

    Examples
    --------
    >>> import ants
    >>> import numpy as np
    >>> fslmat = np.zeros((4,4))
    >>> np.fill_diagonal(fslmat, 1)
    >>> img = ants.image_read(ants.get_ants_data('ch2'))
    >>> tx = ants.fsl2antstransform(fslmat, img, img)
    """
    if reference.dimension != 3:
        raise ValueError('reference image must be 3 dimensions')

    if reference.pixeltype != 'float':
        reference = reference.clone('float')
    if moving.pixeltype != 'float':
        moving = moving.clone('float')

    libfn = get_lib_fn('fsl2antstransformF3')
    tx_ptr = libfn(list(matrix), 
                    reference.pointer,
                    moving.pointer,
                    1)

    return ants.ANTsTransform(precision='float', dimension=reference.dimension, 
                             transform_type='AffineTransform', pointer=tx_ptr)

