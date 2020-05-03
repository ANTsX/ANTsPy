__all__ = ["fit_bspline_object_to_scattered_data"]

import numpy as np

from ..core import ants_image as iio
from .. import utils


def fit_bspline_object_to_scattered_data(scattered_data,
                                         parametric_data,
                                         parametric_domain_origin,
                                         parametric_domain_spacing,
                                         parametric_domain_size,
                                         is_parametric_dimension_closed=None,
                                         data_weights=None,
                                         number_of_fitting_levels=4,
                                         mesh_size=1,
                                         spline_order=3):

    """
    Fit a b-spline object to scattered data.  This is basically a wrapper
    for the ITK filter 
    
    https://itk.org/Doxygen/html/classitk_1_1BSplineScatteredDataPointSetToImageFilter.html
    
    This filter is flexible in the possible objects that can be approximated.
    Possibilities include:

        * 1/2/3/4-D curve
        * 2-D surface in 3-D space (not available/templated)
        * 2/3/4-D scalar field
        * 2/3-D displacement field

    In order to understand the input parameters, it is important to understand
    the difference between the parametric and data dimensions.  A curve as one
    parametric dimension but the data dimension can be 1-D, 2-D, 3-D, or 4-D.
    In contrast, a 3-D displacement field has a parametric and data dimension
    of 3.  The scattered data is what's approximated by the B-spline object and
    the parametric point is the location of scattered data within the domain of
    the B-spline object.

    ANTsR function: `fitBsplineObjectToScatteredData`

    Arguments
    ---------
    scattered_data : 2-D numpy array 
        Defines the scattered data input to be approximated. Data is organized 
        by row --> data v, column ---> data dimension.

    parametric_data : 2-D numpy array 
        Defines the parametric location of the scattered data.  Data is organized 
        by row --> parametric point, column --> parametric dimension.  Note that 
        each row corresponds to the same row in the scatteredData.

    data_weights : 1-D numpy array 
        Defines the individual weighting of the corresponding scattered data value.  
        Default = None meaning all values are weighted the same.

    parametric_domain_origin : n-D tuple 
        Defines the parametric origin of the B-spline object.

    parametric_domain_spacing : n-D tuple 
        Defines the parametric spacing of the B-spline object.  Defines the sampling 
        rate in the parametric domain.

    parametric_domain_size : n-D tuple 
       Defines the size (length) of the B-spline object.  Note that the length of the 
       B-spline object in dimension d is defined as 
       parametric_domain_spacing[d] * parametric_domain_size[d]-1.

    is_parametric_dimension_closed : n-D tuple
       Booleans defining whether or not the corresponding parametric dimension is 
       closed (e.g., closed loop).  Default = None.

    number_of_fitting_levels : integer
       Specifies the number of fitting levels.

    mesh_size : n-D tuple
       Defines the mesh size at the initial fitting level.

    spline_order : integer
       Spline order of the B-spline object.  Default = 3.

    Returns
    -------
    returns numpy array for B-spline curve (parametric dimension = 1).  Otherwise, 
    returns an ANTsImage.

    Example
    -------
    >>> # Perform 2-D curve example
    >>>
    >>> import ants, numpy
    >>> import matplotlib.pyplot as plt
    >>>
    >>> x = numpy.linspace(-4, 4, num=100)
    >>> y = numpy.exp(-numpy.multiply(x, x)) + numpy.random.uniform(-0.1, 0.1, len(x))
    >>> u = numpy.linspace(0, 1.0, num=len(x))
    >>> scattered_data = numpy.column_stack((x, y))
    >>> parametric_data = numpy.expand_dims(u, axis=-1)
    >>> spacing = 1/(len(x)-1) * 1.0;
    >>> 
    >>> bspline_curve = ants.fit_bspline_object_to_scattered_data(scattered_data, parametric_data,
    >>>   parametric_domain_origin=[0.0], parametric_domain_spacing=[spacing],
    >>>   parametric_domain_size=[len(x)], is_parametric_dimension_closed=None,
    >>>   number_of_fitting_levels=5, mesh_size=1)
    >>> 
    >>> plt.plot(x, y, label='Noisy points')
    >>> plt.plot(bspline_curve[:,0], bspline_curve[:,1], label='B-spline curve')
    >>> plt.grid(True)
    >>> plt.axis('tight')
    >>> plt.legend(loc='upper left')
    >>> plt.show()
    >>>
    >>> ###########################################################################
    >>> 
    >>> # Perform 2-D scalar field (i.e., image) example
    >>> 
    >>> import ants, numpy
    >>> 
    >>> number_of_random_points = 10000
    >>> 
    >>> img = ants.image_read( ants.get_ants_data("r16"))
    >>> img_array = img.numpy()
    >>> row_indices = numpy.random.choice(range(2, img_array.shape[0]), number_of_random_points)
    >>> col_indices = numpy.random.choice(range(2, img_array.shape[1]), number_of_random_points)
    >>> 
    >>> scattered_data = numpy.zeros((number_of_random_points, 1))
    >>> parametric_data = numpy.zeros((number_of_random_points, 2))
    >>> 
    >>> for i in range(number_of_random_points):
    >>>     scattered_data[i,0] = img_array[row_indices[i], col_indices[i]]
    >>>     parametric_data[i,0] = row_indices[i]
    >>>     parametric_data[i,1] = col_indices[i]
    >>> 
    >>> bspline_img = ants.fit_bspline_object_to_scattered_data(
    >>>     scattered_data, parametric_data,
    >>>     parametric_domain_origin=[0.0, 0.0], 
    >>>     parametric_domain_spacing=[1.0, 1.0],
    >>>     parametric_domain_size = img.shape, 
    >>>     number_of_fitting_levels=7, mesh_size=1)
    >>> 
    >>> ants.plot(img, title="Original")  
    >>> ants.plot(bspline_img, title="B-spline approximation")  
    """

    parametric_dimension = parametric_data.shape[1]
    data_dimension = scattered_data.shape[1]

    if is_parametric_dimension_closed is None:
        is_parametric_dimension_closed = np.repeat(False, parametric_dimension)

    if isinstance(mesh_size, int) == False and len(mesh_size) != parametric_dimension:
        raise ValueError("Incorrect specification for mesh_size.")

    if len(parametric_domain_origin) != parametric_dimension:
        raise ValueError("Origin is not of length parametric_dimension.")

    if len(parametric_domain_spacing) != parametric_dimension:
        raise ValueError("Spacing is not of length parametric_dimension.")

    if len(parametric_domain_size) != parametric_dimension:
        raise ValueError("Size is not of length parametric_dimension.")

    if len(is_parametric_dimension_closed) != parametric_dimension:
        raise ValueError("Closed is not of length parametric_dimension.")

    number_of_control_points = mesh_size + spline_order

    if isinstance(number_of_control_points, int) == True:
        number_of_control_points = np.repeat(number_of_control_points, parametric_dimension)
 
    if parametric_data.shape[0] != scattered_data.shape[0]:
        raise ValueError("The number of points is not equal to the number of scattered data values.")

    if data_weights is None:
        data_weights = np.repeat(1.0, parametric_data.shape[0])  

    if len(data_weights) != parametric_data.shape[0]:
        raise ValueError("The number of weights is not the same as the number of points.")

    libfn = utils.get_lib_fn("fitBsplineObjectToScatteredDataP%iD%i" % (parametric_dimension, data_dimension))
    bspline_object = libfn(scattered_data, parametric_data, data_weights,
                           parametric_domain_origin, parametric_domain_spacing,
                           parametric_domain_size, is_parametric_dimension_closed,
                           number_of_fitting_levels, number_of_control_points,
                           spline_order)

    if parametric_dimension == 1:
        return bspline_object
    else:
        bspline_image = iio.ANTsImage(pixeltype='float', 
          dimension=parametric_dimension, components=data_dimension,
          pointer=bspline_object).clone('float')
        return bspline_image  

