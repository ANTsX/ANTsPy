__all__ = ["fit_transform_to_paired_points", 
           "fit_time_varying_transform_to_point_sets"]

import numpy as np
import math
import time

import ants

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


def fit_transform_to_paired_points(moving_points,
                                   fixed_points,
                                   transform_type="affine",
                                   regularization=1e-6,
                                   domain_image=None,
                                   number_of_fitting_levels=4,
                                   mesh_size=1,
                                   spline_order=3,
                                   enforce_stationary_boundary=True,
                                   displacement_weights=None,
                                   number_of_compositions=10,
                                   composition_step_size=0.5,
                                   sigma=0.0,
                                   convergence_threshold=1e-6,
                                   number_of_time_steps=2,
                                   number_of_integration_steps=100,
                                   rasterize_points=False,
                                   verbose=False
                                  ):
    """
    Estimate a transform from corresponding fixed and moving landmarks.

    ANTsR function: fitTransformToPairedPoints

    Arguments
    ---------
    moving_points : array
        Moving points specified in physical space as a n x d matrix where n is the number
        of points and d is the dimensionality.

    fixed_points : array
        Fixed points specified in physical space as a n x d matrix where n is the number
        of points and d is the dimensionality.

    transform_type : character
        'rigid', 'similarity', "affine', 'bspline', 'tps', 'diffeo', 'syn', or 'time-varying (tv)'.

    regularization : scalar
        Ridge penalty in [0,1] for linear transforms.

    domain_image : ANTs image
        Defines physical domain of the nonlinear transform.  Must be defined for nonlinear
        transforms.

    number_of_fitting_levels : integer
        Integer specifying the number of fitting levels for the B-spline interpolation of the
        displacement field.

    mesh_size : integer or array
        Defines the mesh size at the initial fitting level for the B-spline interpolation of the
        displacement field.

    spline_order : integer
        Spline order of the B-spline displacement field.

    enforce_stationary_boundary : boolean
        Ensure no displacements on the image boundary (B-spline only).

    displacement_weights : array
        Defines the individual weighting of the corresponding scattered data value.  Default = NULL
        meaning all displacements are weighted the same.

    number_of_compositions : integer
        Total number of compositions for the diffeomorphic transforms.

    composition_step_size : scalar
        Scalar multiplication factor of the weighting of the update field for the diffeomorphic transforms.

    sigma : scalar
        Gaussian smoothing standard deviation of the update field (in mm).

    convergence_threshold : scalar
        Composition-based convergence parameter for the diff. transforms using a
        window size of 10 values.

    number_of_time_steps : integer
        Time-varying velocity field parameter.

    number_of_integration_steps : scalar
        Number of steps used for integrating the velocity field.

    rasterize_points : boolean
       Use nearest neighbor rasterization of points for estimating the update
       field (potential speed-up).  Default = False.

    verbose : bool
        Print progress to the screen.

    Returns
    -------

    ANTs transform

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> fixed = np.array([[50.0,50.0],[200.0,50.0],[200.0,200.0]])
    >>> moving = np.array([[50.0,50.0],[50.0,200.0],[200.0,200.0]])
    >>> xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="affine")
    >>> xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="rigid")
    >>> xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="similarity")
    >>> domain_image = ants.image_read(ants.get_ants_data("r16"))
    >>> xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="bspline", domain_image=domain_image, number_of_fitting_levels=5)
    >>> xfrm = ants.fit_transform_to_paired_points(moving, fixed, transform_type="diffeo", domain_image=domain_image, number_of_fitting_levels=6)
    """

    def polar_decomposition(X):
         U, d, V = np.linalg.svd(X, full_matrices=False)
         P = np.matmul(U, np.matmul(np.diag(d), np.transpose(U)))
         Z = np.matmul(U, V)
         if np.linalg.det(Z) < 0:
             n = X.shape[0]
             reflection_matrix = np.identity(n)
             reflection_matrix[0,0] = -1.0
             Z = np.matmul(Z, reflection_matrix)
         return({"P" : P, "Z" : Z, "Xtilde" : np.matmul(P, Z)})

    def create_zero_displacement_field(domain_image):
         field_array = np.zeros((*domain_image.shape, domain_image.dimension))
         field = ants.from_numpy(field_array, origin=domain_image.origin,
                 spacing=domain_image.spacing, direction=domain_image.direction,
                 has_components=True)
         return(field)

    def create_zero_velocity_field(domain_image, number_of_time_points=2):
         field_array = np.zeros((*domain_image.shape, number_of_time_points, domain_image.dimension))
         origin = (*domain_image.origin, 0.0)
         spacing = (*domain_image.spacing, 1.0)
         direction = np.eye(domain_image.dimension + 1)
         direction[0:domain_image.dimension,0:domain_image.dimension] = domain_image.direction
         field = ants.from_numpy(field_array, origin=origin, spacing=spacing, direction=direction,
                 has_components=True)
         return(field)

    allowed_transforms = ['rigid', 'affine', 'similarity', 'bspline', 'tps', 'diffeo', 'syn', 'tv', 'time-varying']
    if not transform_type.lower() in allowed_transforms:
        raise ValueError(transform_type + " transform not supported.")

    transform_type = transform_type.lower()

    if domain_image is None and transform_type in ['bspline', 'tps', 'diffeo', 'syn', 'tv', 'time-varying']:
        raise ValueError("Domain image needs to be specified.")

    if not fixed_points.shape == moving_points.shape:
        raise ValueError("Mismatch in the size of the point sets.")

    if regularization > 1:
        regularization = 1
    elif regularization < 0:
        regularization = 0

    number_of_points = fixed_points.shape[0]
    dimensionality = fixed_points.shape[1]

    if transform_type in ['rigid', 'affine', 'similarity']:
        center_fixed = fixed_points.mean(axis=0)
        center_moving = moving_points.mean(axis=0)

        x = fixed_points - center_fixed
        y = moving_points - center_moving

        y_prior = np.concatenate((y, np.ones((number_of_points, 1))), axis=1)

        x11 = np.concatenate((x, np.ones((number_of_points, 1))), axis=1)
        M = x11 * (1.0 - regularization) + regularization * y_prior
        Minv = np.linalg.lstsq(M, y, rcond=None)[0]

        p = polar_decomposition(Minv[0:dimensionality, 0:dimensionality].T)
        A = p['Xtilde']
        translation = Minv[dimensionality,:] + center_moving - center_fixed

        if transform_type in ['rigid', 'similarity']:
            # Kabsch algorithm
            #    http://web.stanford.edu/class/cs273/refs/umeyama.pdf

            C = np.dot(y.T, x)
            x_svd = np.linalg.svd(C * (1.0 - regularization) + np.eye(dimensionality) * regularization)
            x_det = np.linalg.det(np.dot(x_svd[0], x_svd[2]))

            if x_det < 0:
                x_svd[2][dimensionality-1, :] *= -1

            A = np.dot(x_svd[0], x_svd[2])

            if transform_type == 'similarity':
                scaling = (math.sqrt((np.power(y, 2).sum(axis=1) / number_of_points).mean()) /
                           math.sqrt((np.power(x, 2).sum(axis=1) / number_of_points).mean()))
                A = np.dot(A, np.eye(dimensionality) * scaling)

        xfrm = ants.create_ants_transform(matrix=A, translation=translation,
              dimension=dimensionality, center=center_fixed)

        return xfrm

    elif transform_type == "bspline":

        bspline_displacement_field = ants.fit_bspline_displacement_field(
            displacement_origins=fixed_points,
            displacements=moving_points - fixed_points,
            displacement_weights=displacement_weights,
            origin=domain_image.origin,
            spacing=domain_image.spacing,
            size=domain_image.shape,
            direction=domain_image.direction,
            number_of_fitting_levels=number_of_fitting_levels,
            mesh_size=mesh_size,
            spline_order=spline_order,
            enforce_stationary_boundary=enforce_stationary_boundary,
            rasterize_points=rasterize_points)

        xfrm = ants.transform_from_displacement_field(bspline_displacement_field)

        return xfrm

    elif transform_type == "tps":

        tps_displacement_field = ants.fit_thin_plate_spline_displacement_field(
            displacement_origins=fixed_points,
            displacements=moving_points - fixed_points,
            origin=domain_image.origin,
            spacing=domain_image.spacing,
            size=domain_image.shape,
            direction=domain_image.direction)

        xfrm = ants.transform_from_displacement_field(tps_displacement_field)

        return xfrm

    elif transform_type == "diffeo":

        if verbose:
            start_total_time = time.time()

        updated_fixed_points = np.empty_like(fixed_points)
        updated_fixed_points[:] = fixed_points

        total_field = create_zero_displacement_field(domain_image)
        total_field_xfrm = None

        error_values = []
        for i in range(number_of_compositions):

            if verbose:
                start_time = time.time()

            update_field = ants.fit_bspline_displacement_field(
              displacement_origins=updated_fixed_points,
              displacements=moving_points - updated_fixed_points,
              displacement_weights=displacement_weights,
              origin=domain_image.origin,
              spacing=domain_image.spacing,
              size=domain_image.shape,
              direction=domain_image.direction,
              number_of_fitting_levels=number_of_fitting_levels,
              mesh_size=mesh_size,
              spline_order=spline_order,
              enforce_stationary_boundary=True,
              rasterize_points=rasterize_points
            )

            update_field = update_field * composition_step_size
            if sigma > 0:
                update_field = ants.smooth_image(update_field, sigma)

            total_field = ants.compose_displacement_fields(update_field, total_field)
            total_field_xfrm = ants.transform_from_displacement_field(total_field)

            if i < number_of_compositions - 1:
                for j in range(updated_fixed_points.shape[0]):
                    updated_fixed_points[j,:] = total_field_xfrm.apply_to_point(tuple(fixed_points[j,:]))

            error_values.append(np.mean(np.sqrt(np.sum(np.square(updated_fixed_points - moving_points), axis=1, keepdims=True))))
            convergence_value = convergence_monitoring(error_values)
            if verbose:
                end_time = time.time()
                diff_time = end_time - start_time
                print("Composition " + str(i) + ": error = " + str(error_values[-1]) +
                      " (convergence = " + str(convergence_value) + ", elapsed time = " + str(diff_time) + ")")
            if not convergence_value is None and convergence_value <= convergence_threshold:
                break

        if verbose:
            end_total_time = time.time()
            diff_total_time = end_total_time - start_total_time
            print("Total elapsed time = " + str(diff_total_time) + ".")

        return(total_field_xfrm)

    elif transform_type == "syn":

        if verbose:
            start_total_time = time.time()

        updated_fixed_points = np.empty_like(fixed_points)
        updated_fixed_points[:] = fixed_points
        updated_moving_points = np.empty_like(moving_points)
        updated_moving_points[:] = moving_points

        total_field_fixed_to_middle = create_zero_displacement_field(domain_image)
        total_inverse_field_fixed_to_middle = create_zero_displacement_field(domain_image)

        total_field_moving_to_middle = create_zero_displacement_field(domain_image)
        total_inverse_field_moving_to_middle = create_zero_displacement_field(domain_image)

        error_values = []
        for i in range(number_of_compositions):

            if verbose:
                start_time = time.time()

            update_field_fixed_to_middle = ants.fit_bspline_displacement_field(
              displacement_origins=updated_fixed_points,
              displacements=updated_moving_points - updated_fixed_points,
              displacement_weights=displacement_weights,
              origin=domain_image.origin,
              spacing=domain_image.spacing,
              size=domain_image.shape,
              direction=domain_image.direction,
              number_of_fitting_levels=number_of_fitting_levels,
              mesh_size=mesh_size,
              spline_order=spline_order,
              enforce_stationary_boundary=True,
              rasterize_points=rasterize_points
            )

            update_field_moving_to_middle = ants.fit_bspline_displacement_field(
              displacement_origins=updated_moving_points,
              displacements=updated_fixed_points - updated_moving_points,
              displacement_weights=displacement_weights,
              origin=domain_image.origin,
              spacing=domain_image.spacing,
              size=domain_image.shape,
              direction=domain_image.direction,
              number_of_fitting_levels=number_of_fitting_levels,
              mesh_size=mesh_size,
              spline_order=spline_order,
              enforce_stationary_boundary=True,
              rasterize_points=rasterize_points
            )

            update_field_fixed_to_middle = update_field_fixed_to_middle * composition_step_size
            update_field_moving_to_middle = update_field_moving_to_middle * composition_step_size
            if sigma > 0:
                update_field_fixed_to_middle = ants.smooth_image(update_field_fixed_to_middle, sigma)
                update_field_moving_to_middle = ants.smooth_image(update_field_moving_to_middle, sigma)

            # Add the update field to both forward displacement fields.

            total_field_fixed_to_middle = ants.compose_displacement_fields(update_field_fixed_to_middle, total_field_fixed_to_middle)
            total_field_moving_to_middle = ants.compose_displacement_fields(update_field_moving_to_middle, total_field_moving_to_middle)

            # Iteratively estimate the inverse fields.

            total_inverse_field_fixed_to_middle = ants.invert_displacement_field(total_field_fixed_to_middle, total_inverse_field_fixed_to_middle)
            total_inverse_field_moving_to_middle = ants.invert_displacement_field(total_field_moving_to_middle, total_inverse_field_moving_to_middle)

            total_field_fixed_to_middle = ants.invert_displacement_field(total_inverse_field_fixed_to_middle, total_field_fixed_to_middle)
            total_field_moving_to_middle = ants.invert_displacement_field(total_inverse_field_moving_to_middle, total_field_moving_to_middle)

            total_field_fixed_to_middle_xfrm = ants.transform_from_displacement_field(total_field_fixed_to_middle)
            total_field_moving_to_middle_xfrm = ants.transform_from_displacement_field(total_field_moving_to_middle)

            total_inverse_field_fixed_to_middle_xfrm = ants.transform_from_displacement_field(total_inverse_field_fixed_to_middle)
            total_inverse_field_moving_to_middle_xfrm = ants.transform_from_displacement_field(total_inverse_field_moving_to_middle)

            if i < number_of_compositions - 1:
                for j in range(updated_fixed_points.shape[0]):
                    updated_fixed_points[j,:] = total_field_fixed_to_middle_xfrm.apply_to_point(tuple(fixed_points[j,:]))
                    updated_moving_points[j,:] = total_field_moving_to_middle_xfrm.apply_to_point(tuple(moving_points[j,:]))

            error_values.append(np.mean(np.sqrt(np.sum(np.square(updated_fixed_points - updated_moving_points), axis=1, keepdims=True))))
            convergence_value = convergence_monitoring(error_values)
            if verbose:
                end_time = time.time()
                diff_time = end_time - start_time
                print("Composition " + str(i) + ": error = " + str(error_values[-1]) +
                      " (convergence = " + str(convergence_value) + ", elapsed time = " + str(diff_time) + ")")
            if not convergence_value is None and convergence_value <= convergence_threshold:
                break

        total_forward_field = ants.compose_displacement_fields(total_inverse_field_moving_to_middle, total_field_fixed_to_middle)
        total_forward_xfrm = ants.transform_from_displacement_field(total_forward_field)
        total_inverse_field = ants.compose_displacement_fields(total_inverse_field_fixed_to_middle, total_field_moving_to_middle)
        total_inverse_xfrm = ants.transform_from_displacement_field(total_inverse_field)

        if verbose:
            end_total_time = time.time()
            diff_total_time = end_total_time - start_total_time
            print("Total elapsed time = " + str(diff_total_time) + ".")

        return_dict = {'forward_transform' : total_forward_xfrm,
                       'inverse_transform' : total_inverse_xfrm,
                       'fixed_to_middle_transform' : total_field_fixed_to_middle_xfrm,
                       'middle_to_fixed_transform' : total_inverse_field_fixed_to_middle_xfrm,
                       'moving_to_middle_transform' : total_field_moving_to_middle_xfrm,
                       'middle_to_moving_transform' : total_inverse_field_moving_to_middle_xfrm
                       }
        return(return_dict)

    elif transform_type == "tv" or transform_type == "time-varying":

        if verbose:
            start_total_time = time.time()

        updated_fixed_points = np.empty_like(fixed_points)
        updated_fixed_points[:] = fixed_points
        updated_moving_points = np.empty_like(moving_points)
        updated_moving_points[:] = moving_points

        velocity_field = create_zero_velocity_field(domain_image, number_of_time_steps)
        velocity_field_array = velocity_field.numpy()

        last_update_derivative_field = create_zero_velocity_field(domain_image, number_of_time_steps)
        last_update_derivative_field_array = last_update_derivative_field.numpy()

        error_values = []
        for i in range(number_of_compositions):

            if verbose:
                start_time = time.time()

            update_derivative_field = create_zero_velocity_field(domain_image, number_of_time_steps)
            update_derivative_field_array = update_derivative_field.numpy()

            average_error = 0.0
            for n in range(number_of_time_steps):

                t = n / (number_of_time_steps - 1.0)

                if n > 0:
                    integrated_forward_field = ants.integrate_velocity_field(velocity_field, 0.0, t, number_of_integration_steps)
                    integrated_forward_field_xfrm = ants.transform_from_displacement_field(integrated_forward_field)
                    for j in range(updated_fixed_points.shape[0]):
                        updated_fixed_points[j,:] = integrated_forward_field_xfrm.apply_to_point(tuple(fixed_points[j,:]))
                else:
                    updated_fixed_points[:] = fixed_points

                if n < number_of_time_steps - 1:
                    integrated_inverse_field = ants.integrate_velocity_field(velocity_field, 1.0, t, number_of_integration_steps)
                    integrated_inverse_field_xfrm = ants.transform_from_displacement_field(integrated_inverse_field)
                    for j in range(updated_moving_points.shape[0]):
                        updated_moving_points[j,:] = integrated_inverse_field_xfrm.apply_to_point(tuple(moving_points[j,:]))
                else:
                    updated_moving_points[:] = moving_points

                update_derivative_field_at_timepoint = ants.fit_bspline_displacement_field(
                  displacement_origins=updated_fixed_points,
                  displacements=updated_moving_points - updated_fixed_points,
                  displacement_weights=displacement_weights,
                  origin=domain_image.origin,
                  spacing=domain_image.spacing,
                  size=domain_image.shape,
                  direction=domain_image.direction,
                  number_of_fitting_levels=number_of_fitting_levels,
                  mesh_size=mesh_size,
                  spline_order=spline_order,
                  enforce_stationary_boundary=True,
                  rasterize_points=rasterize_points
                  )

                if sigma > 0:
                    update_derivative_field_at_timepoint = ants.smooth_image(update_derivative_field_at_timepoint, sigma)

                update_derivative_field_at_timepoint_array = update_derivative_field_at_timepoint.numpy()
                grad_norms = np.sqrt(np.sum(np.square(update_derivative_field_at_timepoint_array), axis=-1, keepdims=False))
                max_norm = np.amax(grad_norms)
                median_norm = np.median(grad_norms)
                if verbose:
                    print("  integration point " + str(t) + ": max_norm = " + str(max_norm) + ", median_norm = " + str(median_norm))
                update_derivative_field_at_timepoint_array /= max_norm
                if domain_image.dimension == 2:
                    update_derivative_field_array[:,:,n,:] = update_derivative_field_at_timepoint_array
                elif domain_image.dimension == 3:
                    update_derivative_field_array[:,:,:,n,:] = update_derivative_field_at_timepoint_array

                rmse = np.mean(np.sqrt(np.sum(np.square(updated_moving_points - updated_fixed_points), axis=1, keepdims=True)))
                average_error = (average_error * n + rmse) / (n + 1)

            update_derivative_field_array = (update_derivative_field_array + last_update_derivative_field_array) * 0.5
            last_update_derivative_field_array = np.empty_like(update_derivative_field_array)
            last_update_derivative_field_array[:] = update_derivative_field_array

            velocity_field_array = velocity_field_array + update_derivative_field_array * composition_step_size
            velocity_field = ants.from_numpy(velocity_field_array, origin=velocity_field.origin,
                                             spacing=velocity_field.spacing, direction=velocity_field.direction,
                                             has_components=True)

            error_values.append(average_error)
            convergence_value = convergence_monitoring(error_values)
            if verbose:
                end_time = time.time()
                diff_time = end_time - start_time
                print("Composition " + str(i) + ": error = " + str(error_values[-1]) +
                      " (convergence = " + str(convergence_value) + ", elapsed time = " + str(diff_time) + ")")
            if not convergence_value is None and convergence_value <= convergence_threshold:
                break

        forward_xfrm = ants.transform_from_displacement_field(ants.integrate_velocity_field(velocity_field, 0.0, 1.0, number_of_integration_steps))
        inverse_xfrm = ants.transform_from_displacement_field(ants.integrate_velocity_field(velocity_field, 1.0, 0.0, number_of_integration_steps))

        if verbose:
            end_total_time = time.time()
            diff_total_time = end_total_time - start_total_time
            print("Total elapsed time = " + str(diff_total_time) + ".")

        return_dict = {'forward_transform': forward_xfrm,
                       'inverse_transform': inverse_xfrm,
                       'velocity_field': velocity_field}
        return(return_dict)

    else:
        raise ValueError("Unrecognized transform_type.")


def fit_time_varying_transform_to_point_sets(point_sets,
                                             time_points=None,
                                             initial_velocity_field=None,
                                             number_of_time_steps=None,
                                             domain_image=None,
                                             number_of_fitting_levels=4,
                                             mesh_size=1,
                                             spline_order=3,
                                             displacement_weights=None,
                                             number_of_compositions=10,
                                             composition_step_size=0.5,
                                             number_of_integration_steps=100,
                                             sigma=0.0,
                                             convergence_threshold=1e-6,
                                             rasterize_points=False,
                                             verbose=False
                                            ):
    """

    Estimate a time-varying transform from corresponding point sets (> 2).

    ANTsR function: fitTimeVaryingTransformToPointSets

    Arguments
    ---------
    point_sets : list of arrays
        Corresponding points across sets specified in physical space as a n x d matrix where n
        is the number of points and d is the dimensionality.

    time_points : array of ordered scalars between 0 and 1
        Set of scalar values, one for each point-set, designating its time position in the velocity
        flow.  If not set, it defaults to equal spacing between 0 and 1.

    initial_velocity_field : initial ANTs velocity field
        Optional velocity field for initializing optimization.  Overrides the number of integration
        points.

    number_of_time_steps : integer
        Time-varying velocity field parameter.  Needs to be equal to or greater than the number of
        point sets.  If not specified, it defaults to the number of point sets.

    domain_image : ANTs image
        Defines physical domain of the nonlinear transform.  Must be defined.

    number_of_fitting_levels : integer
        Integer specifying the number of fitting levels for the B-spline interpolation of the
        displacement field.

    mesh_size : integer or array
        Defines the mesh size at the initial fitting level for the B-spline interpolation of the
        displacement field..

    spline_order : integer
        Spline order of the B-spline displacement field.

    displacement_weights : array
        Defines the individual weighting of the corresponding scattered data value.  Default = NULL
        meaning all displacements are weighted the same.

    number_of_compositions : integer
        Total number of compositions.

    composition_step_size : scalar
        Scalar multiplication factor of the weighting of the update field.

    number_of_integration_steps : scalar
        Number of steps used for integrating the velocity field.

    sigma : scalar
        Gaussian smoothing standard deviation of the update field (in mm).

    convergence_threshold : scalar
        Composition-based convergence parameter using a window size of 10 values.

    rasterize_points : boolean
        Use nearest neighbor rasterization of points for estimating the update field (potential
        speed-up).  Default = False.

    verbose : bool
        Print progress to the screen.

    Returns
    -------

    ANTs transform

    Example
    -------
    >>> import ants
    >>> import numpy as np
    """

    def create_zero_velocity_field(domain_image, number_of_time_points=2):
         field_array = np.zeros((*domain_image.shape, number_of_time_points, domain_image.dimension))
         origin = (*domain_image.origin, 0.0)
         spacing = (*domain_image.spacing, 1.0)
         direction = np.eye(domain_image.dimension + 1)
         direction[0:domain_image.dimension,0:domain_image.dimension] = domain_image.direction
         field = ants.from_numpy(field_array, origin=origin, spacing=spacing, direction=direction,
                 has_components=True)
         return(field)

    if not isinstance(point_sets, list):
        raise ValueError("point_sets should be a list of corresponding point sets.")

    number_of_point_sets = len(point_sets)

    if time_points is not None and len(time_points) != number_of_point_sets:
        raise ValueError("The number of time points should be the same as the number of point sets.")

    if time_points is None:
        time_points = np.linspace(0.0, 1.0, number_of_point_sets)
    time_points = np.array(time_points)

    if np.any(time_points < 0.0) or np.any(time_points > 1.0):
        raise ValueError("time point values should be between 0 and 1.")

    if number_of_point_sets < 3:
        raise ValueError("Expecting three or greater point sets.")

    if domain_image is None:
        raise ValueError("Domain image needs to be specified.")

    number_of_points = point_sets[0].shape[0]
    dimensionality = point_sets[0].shape[1]
    for i in range(1, number_of_point_sets):
        if point_sets[i].shape[0] != number_of_points:
            raise ValueError("Point sets should match in terms of the number of points.")
        if point_sets[i].shape[1] != dimensionality:
            raise ValueError("Point sets should match in terms of dimensionality.")

    if verbose:
        start_total_time = time.time()

    updated_fixed_points = np.zeros(point_sets[0].shape)
    updated_moving_points = np.zeros(point_sets[0].shape)

    velocity_field = None
    if initial_velocity_field is None:
        if number_of_time_steps is None:
            number_of_time_steps = len(time_points)
        if number_of_time_steps < number_of_point_sets:
            raise ValueError("The number of integration points should be at least as great as the number of point sets.")
        velocity_field = create_zero_velocity_field(domain_image, number_of_time_steps)
    else:
        velocity_field = ants.image_clone(initial_velocity_field)
        number_of_time_steps = initial_velocity_field.shape[-1]
    velocity_field_array = velocity_field.numpy()

    last_update_derivative_field = create_zero_velocity_field(domain_image, number_of_time_steps)
    last_update_derivative_field_array = last_update_derivative_field.numpy()

    error_values = []
    for i in range(number_of_compositions):

        if verbose:
            start_time = time.time()

        update_derivative_field = create_zero_velocity_field(domain_image, number_of_time_steps)
        update_derivative_field_array = update_derivative_field.numpy()

        average_error = 0.0
        for n in range(number_of_time_steps):

            t = n / (number_of_time_steps - 1.0)

            t_index = 0
            for j in range(1, number_of_point_sets):
                if time_points[j-1] <= t and time_points[j] >= t:
                    t_index = j
                    break

            if n > 0 and n < number_of_time_steps - 1 and time_points[t_index-1] == t:
                updated_fixed_points[:] = point_sets[t_index-1]
                integrated_inverse_field = ants.integrate_velocity_field(velocity_field, time_points[t_index], t, number_of_integration_steps)
                integrated_inverse_field_xfrm = ants.transform_from_displacement_field(integrated_inverse_field)
                for j in range(updated_moving_points.shape[0]):
                    updated_moving_points[j,:] = integrated_inverse_field_xfrm.apply_to_point(tuple(point_sets[t_index][j,:]))

                update_derivative_field_at_timepoint_forward = ants.fit_bspline_displacement_field(
                  displacement_origins=updated_fixed_points,
                  displacements=updated_moving_points - updated_fixed_points,
                  displacement_weights=displacement_weights,
                  origin=domain_image.origin,
                  spacing=domain_image.spacing,
                  size=domain_image.shape,
                  direction=domain_image.direction,
                  number_of_fitting_levels=number_of_fitting_levels,
                  mesh_size=mesh_size,
                  spline_order=spline_order,
                  enforce_stationary_boundary=True,
                  rasterize_points=rasterize_points
                  )

                updated_moving_points[:] = point_sets[t_index-1]
                integrated_forward_field = ants.integrate_velocity_field(velocity_field, time_points[t_index-2], t, number_of_integration_steps)
                integrated_forward_field_xfrm = ants.transform_from_displacement_field(integrated_forward_field)
                for j in range(updated_fixed_points.shape[0]):
                    updated_fixed_points[j,:] = integrated_forward_field_xfrm.apply_to_point(tuple(point_sets[t_index-2][j,:]))

                update_derivative_field_at_timepoint_back = ants.fit_bspline_displacement_field(
                  displacement_origins=updated_fixed_points,
                  displacements=updated_moving_points - updated_fixed_points,
                  displacement_weights=displacement_weights,
                  origin=domain_image.origin,
                  spacing=domain_image.spacing,
                  size=domain_image.shape,
                  direction=domain_image.direction,
                  number_of_fitting_levels=number_of_fitting_levels,
                  mesh_size=mesh_size,
                  spline_order=spline_order,
                  enforce_stationary_boundary=True,
                  rasterize_points=rasterize_points
                  )

                update_derivative_field_at_timepoint = (update_derivative_field_at_timepoint_forward +
                                                        update_derivative_field_at_timepoint_back) / 2.0

            else:
                if t == 0.0 and time_points[t_index-1] == 0.0:
                    updated_fixed_points[:] = point_sets[0]
                else:
                    integrated_forward_field = ants.integrate_velocity_field(velocity_field, time_points[t_index-1], t, number_of_integration_steps)
                    integrated_forward_field_xfrm = ants.transform_from_displacement_field(integrated_forward_field)
                    for j in range(updated_fixed_points.shape[0]):
                        updated_fixed_points[j,:] = integrated_forward_field_xfrm.apply_to_point(tuple(point_sets[t_index-1][j,:]))

                if t == 1.0 and time_points[t_index] == 1.0:
                    updated_moving_points[:] = point_sets[-1]
                else:
                    integrated_inverse_field = ants.integrate_velocity_field(velocity_field, time_points[t_index], t, number_of_integration_steps)
                    integrated_inverse_field_xfrm = ants.transform_from_displacement_field(integrated_inverse_field)
                    for j in range(updated_moving_points.shape[0]):
                        updated_moving_points[j,:] = integrated_inverse_field_xfrm.apply_to_point(tuple(point_sets[t_index][j,:]))

                update_derivative_field_at_timepoint = ants.fit_bspline_displacement_field(
                  displacement_origins=updated_fixed_points,
                  displacements=updated_moving_points - updated_fixed_points,
                  displacement_weights=displacement_weights,
                  origin=domain_image.origin,
                  spacing=domain_image.spacing,
                  size=domain_image.shape,
                  direction=domain_image.direction,
                  number_of_fitting_levels=number_of_fitting_levels,
                  mesh_size=mesh_size,
                  spline_order=spline_order,
                  enforce_stationary_boundary=True,
                  rasterize_points=rasterize_points
                  )

            if sigma > 0:
                update_derivative_field_at_timepoint = ants.smooth_image(update_derivative_field_at_timepoint, sigma)

            update_derivative_field_at_timepoint_array = update_derivative_field_at_timepoint.numpy()
            grad_norms = np.sqrt(np.sum(np.square(update_derivative_field_at_timepoint_array), axis=-1, keepdims=False))
            max_norm = np.amax(grad_norms)
            median_norm = np.median(grad_norms)
            if verbose:
                print("  integration point " + str(t) + ": max_norm = " + str(max_norm) + ", median_norm = " + str(median_norm))
            update_derivative_field_at_timepoint_array /= max_norm
            if domain_image.dimension == 2:
                update_derivative_field_array[:,:,n,:] = update_derivative_field_at_timepoint_array
            elif domain_image.dimension == 3:
                update_derivative_field_array[:,:,:,n,:] = update_derivative_field_at_timepoint_array

            rmse = np.mean(np.sqrt(np.sum(np.square(updated_moving_points - updated_fixed_points), axis=1, keepdims=True)))
            average_error = (average_error * n + rmse) / (n + 1)

        update_derivative_field_array = (update_derivative_field_array + last_update_derivative_field_array) * 0.5
        last_update_derivative_field_array = np.empty_like(update_derivative_field_array)
        last_update_derivative_field_array[:] = update_derivative_field_array

        velocity_field_array += (update_derivative_field_array * composition_step_size)
        velocity_field = ants.from_numpy(velocity_field_array, origin=velocity_field.origin,
                                         spacing=velocity_field.spacing, direction=velocity_field.direction,
                                         has_components=True)

        error_values.append(average_error)
        convergence_value = convergence_monitoring(error_values)
        if verbose:
            end_time = time.time()
            diff_time = end_time - start_time
            print("Composition " + str(i) + ": error = " + str(error_values[-1]) +
                  " (convergence = " + str(convergence_value) + ", elapsed time = " + str(diff_time) + ")")
        if not convergence_value is None and convergence_value <= convergence_threshold:
            break

    forward_xfrm = ants.transform_from_displacement_field(ants.integrate_velocity_field(velocity_field, 0.0, 1.0, number_of_integration_steps))
    inverse_xfrm = ants.transform_from_displacement_field(ants.integrate_velocity_field(velocity_field, 1.0, 0.0, number_of_integration_steps))

    if verbose:
        end_total_time = time.time()
        diff_total_time = end_total_time - start_total_time
        print("Total elapsed time = " + str(diff_total_time) + ".")

    return_dict = {'forward_transform': forward_xfrm,
                   'inverse_transform': inverse_xfrm,
                   'velocity_field': velocity_field}
    return(return_dict)




