__all__ = ["fit_thin_plate_spline_displacement_field"]

import numpy as np

import ants
from ants.internal import get_lib_fn


def fit_thin_plate_spline_displacement_field(displacement_origins=None,
                                             displacements=None,
                                             origin=None,
                                             spacing=None,
                                             size=None,
                                             direction=None):

    """
    Fit a thin-plate spline object to a a set of points with associated displacements.  
    This is basically a wrapper for the ITK filter

    https://itk.org/Doxygen/html/itkThinPlateSplineKernelTransform_8h.html

    ANTsR function: `fitThinPlateSplineToDisplacementField`

    Arguments
    ---------

    displacement_origins : 2-D numpy array
        Matrix (number_of_points x dimension) defining the origins of the input
        displacement points.  Default = None.

    displacements : 2-D numpy array
        Matrix (number_of_points x dimension) defining the displacements of the input
        displacement points.  Default = None.

    origin : n-D tuple
        Defines the physical origin of the B-spline object.

    spacing : n-D tuple
        Defines the physical spacing of the B-spline object.

    size : n-D tuple
       Defines the size (length) of the spline object.  Note that the length of the
       spline object in dimension d is defined as spacing[d] * size[d]-1.

    direction : 2-D numpy array
       Booleans defining whether or not the corresponding parametric dimension is
       closed (e.g., closed loop).  Default = None.

    Returns
    -------
    Returns an ANTsImage.

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> points = np.array([[-50, -50]])
    >>> deltas = np.array([[10, 10]])
    >>> tps_field = ants.fit_thin_plate_spline_displacement_field(
    >>>   displacement_origins=points, displacements=deltas,
    >>>   origin=[0.0, 0.0], spacing=[1.0, 1.0], size=[100, 100],
    >>>   direction=np.array([[-1, 0], [0, -1]]))
    """

    dimensionality = displacement_origins.shape[1]
    if displacements.shape[1] != dimensionality:
        raise ValueError("Dimensionality between origins and displacements does not match.")

    if displacement_origins is None or displacement_origins is None:
        raise ValueError("Missing input.  Input point set (origins + displacements) needs to be specified." )

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

    tps_field = None
    libfn = get_lib_fn("fitThinPlateSplineDisplacementFieldToScatteredDataD%i" % (dimensionality))
    tps_field = libfn(displacement_origins, displacements, origin, spacing, size, direction)

    tps_displacement_field = ants.from_pointer(tps_field).clone('float')
    return tps_displacement_field

