import numpy as np
import ants

__all__ = ['convergence_monitoring']

def convergence_monitoring(values, window_size=10):

     if len(values) >= window_size:

         u = np.linspace(0.0, 1.0, num=window_size)
         scattered_data = np.expand_dims(values[-window_size:], axis=-1)
         parametric_data = np.expand_dims(u, axis=-1)
         spacing = 1 / (window_size-1)
         bspline_line = ants.fit_bspline_object_to_scattered_data(scattered_data, parametric_data,
             parametric_domain_origin=[0.0], parametric_domain_spacing=[spacing],
             parametric_domain_size=[window_size], number_of_fitting_levels=1, mesh_size=1,
             spline_order=1)
         bspline_slope = -(bspline_line[1][0] - bspline_line[0][0]) / spacing

         return(bspline_slope)

     else:

         return None

