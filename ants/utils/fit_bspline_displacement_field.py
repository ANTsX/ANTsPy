__all__ = ["fit_bspline_displacement_field"]

import numpy as np

from ..core import ants_image as iio
from .. import core
from .. import utils


def fit_bspline_displacement_field(displacement_field=None,
                                   displacement_weight_image=None,
                                   displacement_origins=None,
                                   displacements=None,
                                   displacement_weights=None,
                                   origin=None,
                                   spacing=None,
                                   size=None,
                                   direction=None,
                                   number_of_fitting_levels=4,
                                   mesh_size=1,
                                   spline_order=3,
                                   enforce_stationary_boundary=True,
                                   estimate_inverse=False):

    """
    Fit a b-spline object to a dense displacement field image and/or a set of points
    with associated displacements and smooths them using B-splines.  The inverse
    can also be estimated..  This is basically a wrapper for the ITK filter

    https://itk.org/Doxygen/html/classitk_1_1DisplacementFieldToBSplineImageFilter.html}

    which, in turn is a wrapper for the ITK filter used for the function
    fit_bspline_object_to_scattered_data.

    ANTsR function: `fitBsplineToDisplacementField`

    Arguments
    ---------
    displacement_field : ANTs image
        Input displacement field.  Either this and/or the points must be specified.

    displacement_weight_image : ANTs image
        Input image defining weighting of the voxelwise displacements in the displacement_field.  I
        If None, defaults to identity weighting for each displacement.  Default = None.

    displacement_origins : 2-D numpy array
        Matrix (number_of_points x dimension) defining the origins of the input
        displacement points.  Default = None.

    displacements : 2-D numpy array
        Matrix (number_of_points x dimension) defining the displacements of the input
        displacement points.  Default = None.

    displacement_weights : 1-D numpy array
        Array defining the individual weighting of the corresponding scattered data value.
        Default = None meaning all values are weighted the same.

    origin : n-D tuple
        Defines the physical origin of the B-spline object.

    spacing : n-D tuple
        Defines the physical spacing of the B-spline object.

    size : n-D tuple
       Defines the size (length) of the B-spline object.  Note that the length of the
       B-spline object in dimension d is defined as
       spacing[d] * size[d]-1.

    direction : 2-D numpy array
       Booleans defining whether or not the corresponding parametric dimension is
       closed (e.g., closed loop).  Default = None.

    number_of_fitting_levels : integer
       Specifies the number of fitting levels.

    mesh_size : n-D tuple
       Defines the mesh size at the initial fitting level.

    spline_order : integer
       Spline order of the B-spline object.  Default = 3.

    enforce_stationary_boundary : boolean
       Ensure no displacements on the image boundary.  Default = True.

    estimate_inverse : boolean
       Estimate the inverse displacement field.  Default = False.

    Returns
    -------
    Returns an ANTsImage.

    Example
    -------
    >>> # Perform 2-D fitting
    >>>
    >>> import ants, numpy
    >>>
    >>> points = numpy.array([[-50, -50]])
    >>> deltas = numpy.array([[10, 10]])
    >>>
    >>> bspline_field = ants.fit_bspline_displacement_field(
    >>>   displacement_origins=points, displacements=deltas,
    >>>   origin=[0.0], spacing=[spacing], size=[100, 100],
    >>>   direction=numpy.array([[-1, 0], [0, -1]]),
    >>>   number_of_fitting_levels=4, mesh_size=(1, 1))
    """

    if displacement_field is None and (displacement_origins is None or displacements is None):
        raise ValueError("Missing input.  Either a displacement field or input point set (origins + displacements) needs to be specified.")

    if displacement_field is None:
        if origin is None or spacing is None or size is None or direction is None:
            raise ValueError("If the displacement field is not specified, one must fully specify the input physical domain.")

    if displacement_field is not None and displacement_weight_image is None:
        displacement_weight_image = core.make_image(displacement_field.shape, voxval=1,
            spacing=displacement_field.spacing, origin=displacement_field.origin,
            direction=displacement_field.direction, has_components=False, pixeltype='float')

    if displacement_field is not None:
        if origin is None:
            origin = displacement_field.origin
        if spacing is None:
            spacing = displacement_field.spacing
        if direction is None:
            direction = displacement_field.direction
        if size is None:
            size = displacement_field.shape

    dimensionality = None
    if displacement_field is not None:
        dimensionality = displacement_field.dimension
    else:
        dimensionality = displacement_origins.shape[1]
        if displacements.shape[1] != dimensionality:
            raise ValueError("Dimensionality between origins and displacements does not match.")

    if displacement_origins is not None:
        if displacement_weights is not None and (len(displacement_weights) != displacement_origins.shape[0]):
            raise ValueError("Length of displacement weights must match the number of displacement points.")
        else:
            displacement_weights = np.ones(displacement_origins.shape[0])

    if isinstance(mesh_size, int) == False and len(mesh_size) != dimensionality:
        raise ValueError("Incorrect specification for mesh_size.")

    if origin is not None and len(origin) != dimensionality:
        raise ValueError("Origin is not of length dimensionality.")

    if spacing is not None and len(spacing) != dimensionality:
        raise ValueError("Spacing is not of length dimensionality.")

    if size is not None and len(size) != dimensionality:
        raise ValueError("Size is not of length dimensionality.")

    if direction is not None and (direction.shape[0] != dimensionality and direction.shape[1] != dimensionality):
        raise ValueError("Direction is not of shape dimensionality x dimensionality.")

    # It would seem that pybind11 doesn't really play nicely when the
    # arguments are 'None'

    if origin is None:
        origin = np.empty(0)

    if spacing is None:
        spacing = np.empty(0)

    if size is None:
        size = np.empty(0)

    if direction is None:
        direction = np.empty((0, 0))

    if displacement_origins is None:
        displacement_origins = np.empty((0, 0))

    if displacements is None:
        displacements = np.empty((0, 0))

    if displacement_weights is None:
        displacement_weights = np.empty(0)

    number_of_control_points = list(np.array(mesh_size) + np.repeat(spline_order, dimensionality))

    bspline_field = None
    if displacement_field is not None:
        libfn = utils.get_lib_fn("fitBsplineDisplacementFieldD%i" % (dimensionality))
        bspline_field = libfn(displacement_field.pointer, displacement_weight_image.pointer,
                              displacement_origins, displacements, displacement_weights,
                              origin, spacing, size, direction,
                              number_of_fitting_levels, number_of_control_points, spline_order,
                              enforce_stationary_boundary, estimate_inverse)
    elif displacement_field is None and displacements is not None:
        libfn = utils.get_lib_fn("fitBsplineDisplacementFieldToScatteredDataD%i" % (dimensionality))
        bspline_field = libfn(displacement_origins, displacements, displacement_weights,
                              origin, spacing, size, direction,
                              number_of_fitting_levels, number_of_control_points, spline_order,
                              enforce_stationary_boundary, estimate_inverse)


    bspline_displacement_field = iio.ANTsImage(pixeltype='float',
        dimension=dimensionality, components=dimensionality,
        pointer=bspline_field).clone('float')
    return bspline_displacement_field

