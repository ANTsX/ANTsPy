__all__ = ["fit_transform_to_paired_points"]

import os

import numpy as np
from ..core import ants_transform_io as txio
from ..utils import fit_bspline_displacement_field
from ..utils import smooth_image

def fit_transform_to_paired_points( moving_points,
                                    fixed_points,
                                    transform_type="affine",
                                    regularization=1e-6,
                                    domain_image = None,
                                    number_of_fitting_levels=4,
                                    mesh_size=1,
                                    spline_order=3,
                                    enforce_stationary_boundary=True,
                                    displacement_weights=None,
                                    number_of_compositions=10,
                                    composition_step_size=0.5,
                                    sigma=3.0
                                   ):
    """
    Estimate an optimal matrix transformation from paired points, potentially landmarks

    ANTsR function: fitTransformToPairedPoints

    Arguments
    ---------
    moving_points : array
        moving points specified in physical space as a n x d matrix where n is the number
        of points and \code{d} is the dimensionality.

    fixed_points : array
        fixed points specified in physical space as a n x d matrix where n is the number
        of points and \code{d} is the dimensionality.

    transform_type : character
        'rigid', 'similarity', "affine', 'bspline', or 'diffeo'.

    regularization : scalar
        ridge penalty in [0,1] for linear transforms.

    domain_image : ANTs image
        defines physical domain of the B-spline transform.

    number_of_fitting_levels : integer
        integer specifying the number of fitting levels (B-spline only).

    mesh_size : integer or array
        defining the mesh size at the initial fitting level (B-spline only).

    spline_order : integer
        spline order of the B-spline object. (B-spline only).

    enforce_stationary_boundary : boolean
        ensure no displacements on the image boundary (B-spline only).

    displacement_weights : array
        defines the individual weighting of the corresponding scattered data value.  (B-spline
        only).  Default = NULL meaning all displacements are weighted the same.

    number_of_compositions : integer
        total number of compositions for the diffeomorphic transform.

    composition_step_size : scalar
        scalar multiplication factor for the diffeomorphic transform.

    sigma : scalar
        gaussian smoothing sigma (in mm) for the diffeomorphic transform.

    Returns
    -------

    ANTs transform

    Example
    -------
    >>> import ants
    >>> import numpy as np
    >>> fixed = np.array([[50,50],[200,50],[50,200]])
    >>> moving = np.array([[75,75],[175,75],[75,175]])
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

    allowed_transforms = ['rigid', 'affine', 'similarity', 'bspline', 'diffeo']
    if not transform_type.lower() in allowed_transforms:
        raise ValueError(transform_type + " transform not supported.")

    transform_type = transform_type.lower()

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
                scaling = (np.math.sqrt((np.power(y, 2).sum(axis=1) / number_of_points).mean()) /
                           np.math.sqrt((np.power(x, 2).sum(axis=1) / number_of_points).mean()))
                A = np.dot(A, np.eye(dimensionality) * scaling)

        xfrm = txio.create_ants_transform(matrix=A, translation=translation,
              dimension=dimensionality, center=center_fixed)

        return xfrm

    elif transform_type == "diffeo":

        bspline_displacement_field = fit_bspline_displacement_field(
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
            enforce_stationary_boundary=enforce_stationary_boundary)

        xfrm = txio.transform_from_displacement_field(bspline_displacement_field)

        return xfrm

    else:

        updated_fixed_points = fixed_points

        xfrm_list = []
        total_field_xfrm = None

        for i in range(number_of_compositions):

            update_field = fit_bspline_displacement_field(
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
              enforce_stationary_boundary=True
            )

            update_field = update_field * composition_step_size
            if sigma > 0:
                update_field = smooth_image(update_field, sigma)

            xfrm_list.append(txio.transform_from_displacement_field(update_field))
            total_field_xfrm = txio.compose_ants_transform(xfrm_list)

            if i < number_of_compositions - 1:
                for j in range(updated_fixed_points.shape[0]):
                    updated_fixed_points[j,:] = total_field_xfrm.apply_ants_transform_to_point(tuple(updated_fixed_points[j,:]))

            return(total_field_xfrm)




