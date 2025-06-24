__all__ = ["randomly_transform_image_data"]

import ants
import numpy as np

def randomly_transform_image_data(reference_image,
                                  input_image_list,
                                  segmentation_image_list=None,
                                  number_of_simulations=10,
                                  transform_type='affine',
                                  sd_affine=0.02,
                                  deformation_transform_type="bspline",
                                  number_of_random_points=1000,
                                  sd_noise=10.0,
                                  number_of_fitting_levels=4,
                                  mesh_size=1,
                                  sd_smoothing=4.0,
                                  input_image_interpolator='linear',
                                  segmentation_image_interpolator='nearestNeighbor'):

    """
    Randomly transform image data (optional: with corresponding segmentations).

    Apply rigid, affine and/or deformable maps to an input set of training
    images.  The reference image domain defines the space in which this
    happens.

    Arguments
    ---------
    reference_image : ANTsImage
        Defines the spatial domain for all output images.  If the input images do
        not match the spatial domain of the reference image, we internally
        resample the target to the reference image.  This could have unexpected
        consequences.  Resampling to the reference domain is performed by testing
        using ants.image_physical_space_consistency then calling
        ants.resample_image_to_target with failure.

    input_image_list : list of lists of ANTsImages
        List of lists of input images to warp.  The internal list sets contain one
        or more images (per subject) which are assumed to be mutually aligned.  The
        outer list contains multiple subject lists which are randomly sampled to
        produce output image list.

    segmentation_image_list : list of ANTsImages
        List of segmentation images corresponding to the input image list (optional).

    number_of_simulations : integer
        Number of simulated output image sets.

    transform_type : string
        One of the following options: "translation", "rotation", "rigid", "scaleShear", 
        "affine", "deformation", "affineAndDeformation".

    sd_affine : float
        Parameter dictating deviation amount from identity for random linear
        transformations.

    deformation_transform_type : string
        "bspline" or "exponential".

    number_of_random_points : integer
        Number of displacement points for the deformation field.

    sd_noise : float
        Standard deviation of the displacement field.

    number_of_fitting_levels : integer
        Number of fitting levels (bspline deformation only).

    mesh_size : int or n-D tuple
        Determines fitting resolution (bspline deformation only).

    sd_smoothing : float
        Standard deviation of the Gaussian smoothing in mm (exponential field only).

    input_image_interpolator : string
        One of the following options: "nearestNeighbor", "linear", "gaussian", "bSpline".

    segmentation_image_interpolator : string
        Only "nearestNeighbor" is currently available.

    Returns
    -------
    list of lists of transformed images

    Example
    -------
    >>> import ants
    >>> image1_list = list()
    >>> image1_list.append(ants.image_read(ants.get_ants_data("r16")))
    >>> image2_list = list()
    >>> image2_list.append(ants.image_read(ants.get_ants_data("r64")))
    >>> input_segmentations = list()
    >>> input_segmentations.append(ants.threshold_image(image1, "Otsu", 3))
    >>> input_segmentations.append(ants.threshold_image(image2, "Otsu", 3))
    >>> input_images = list()
    >>> input_images.append(image1_list)
    >>> input_images.append(image2_list)
    >>> data = antspynet.randomly_transform_image_data(image1,
    >>>     input_images, input_segmentations, sd_affine=0.02,
    >>>     transform_type = "affineAndDeformation" )
    """

    def create_random_linear_transform(image,
                                       fixed_parameters,
                                       transform_type='affine',
                                       sd_affine=0.02):
        transform = ants.create_ants_transform(transform_type=
           "AffineTransform", precision='float', dimension=image.dimension)
        ants.set_ants_transform_fixed_parameters(transform, fixed_parameters)
        identity_parameters = ants.get_ants_transform_parameters(transform)
        random_epsilon = np.random.normal(loc=0, scale=sd_affine, size=len(identity_parameters))

        if transform_type == 'translation':
            random_epsilon[:(len(identity_parameters) - image.dimension)] = 0
        elif transform_type == "rotation": 
            random_epsilon[(len(identity_parameters) - image.dimension):] = 0

        random_parameters = identity_parameters + random_epsilon
        random_matrix = np.reshape(
            random_parameters[:(len(identity_parameters) - image.dimension)],
            newshape=(image.dimension, image.dimension))
        decomposition = ants.polar_decomposition(random_matrix)

        if transform_type == "rotation" or transform_type == "rigid":
            random_matrix = decomposition['Z']
        elif transform_type == "affine":
            random_matrix = decomposition['Xtilde']
        elif transform_type == "scaleShear":
            random_matrix = decomposition['P']

        random_parameters[:(len(identity_parameters) - image.dimension)] = \
            np.reshape(random_matrix, newshape=(len(identity_parameters) - image.dimension))
        ants.set_ants_transform_parameters(transform, random_parameters)
        return(transform)

    def create_random_displacement_field_transform(image,
                                                   field_type="bspline",
                                                   number_of_random_points=1000,
                                                   sd_noise=10.0,
                                                   number_of_fitting_levels=4,
                                                   mesh_size=1,
                                                   sd_smoothing=4.0):
        displacement_field = ants.simulate_displacement_field(image,
            field_type=field_type, number_of_random_points=number_of_random_points,
            sd_noise=sd_noise, enforce_stationary_boundary=True,
            number_of_fitting_levels=number_of_fitting_levels, mesh_size=mesh_size,
            sd_smoothing=sd_smoothing)
        return(ants.transform_from_displacement_field(displacement_field))

    admissible_transforms = ("translation", "rotation", "rigid", "scaleShear", "affine",
        "affineAndDeformation", "deformation")
    if not transform_type in admissible_transforms:
        raise ValueError("The specified transform is not a possible option.  Please see help menu.")

    # Get the fixed parameters from the reference image.

    fixed_parameters = ants.get_center_of_mass(reference_image)
    number_of_subjects = len(input_image_list)

    random_indices = np.random.choice(number_of_subjects, size=number_of_simulations, replace=True)

    simulated_image_list = list()
    simulated_segmentation_image_list = list()
    simulated_transforms = list()

    for i in range(number_of_simulations):
        single_subject_image_list = input_image_list[random_indices[i]]
        single_subject_segmentation_image = None
        if segmentation_image_list is not None:
            single_subject_segmentation_image = segmentation_image_list[random_indices[i]]

        if ants.image_physical_space_consistency(reference_image, single_subject_image_list[0]) is False:
            for j in range(len(single_subject_image_list)):
                single_subject_image_list.append(
                    ants.resample_image_to_target(single_subject_image_list[j], reference_image,
                        interp_type=input_image_interpolator))
            if single_subject_segmentation_image is not None:
                single_subject_segmentation_image = \
                    ants.resample_image_to_target(single_subject_segmentation_image, reference_image,
                        interp_type=segmentation_image_interpolator)

        transforms = list()

        if transform_type == 'deformation':
            deformable_transform = create_random_displacement_field_transform(
                reference_image, deformation_transform_type, number_of_random_points,
                sd_noise, number_of_fitting_levels, mesh_size, sd_smoothing)
            transforms.append(deformable_transform)
        elif transform_type == 'affineAndDeformation':
            deformable_transform = create_random_displacement_field_transform(
                reference_image, deformation_transform_type, number_of_random_points,
                sd_noise, number_of_fitting_levels, mesh_size, sd_smoothing)
            linear_transform = create_random_linear_transform(reference_image,
                fixed_parameters, 'affine', sd_affine)
            transforms.append(deformable_transform)
            transforms.append(linear_transform)
        else:
            linear_transform = create_random_linear_transform(reference_image,
                fixed_parameters, transform_type, sd_affine)
            transforms.append(linear_transform)

        simulated_transforms.append(ants.compose_ants_transforms(transforms))

        single_subject_simulated_image_list = list()
        for j in range(len(single_subject_image_list)):
            single_subject_image = single_subject_image_list[j]
            single_subject_simulated_image_list.append(ants.apply_ants_transform_to_image(
                simulated_transforms[i], single_subject_image,
                reference=reference_image, interpolation=input_image_interpolator.lower()))

        simulated_image_list.append(single_subject_simulated_image_list)

        if single_subject_segmentation_image is not None:
            simulated_segmentation_image_list.append(ants.apply_ants_transform_to_image(
                simulated_transforms[i], single_subject_segmentation_image,
                reference=reference_image, interpolation=segmentation_image_interpolator.lower()))

    if segmentation_image_list is None:
        return({'simulated_images' : simulated_image_list,
                'simulated_transforms' : simulated_transforms,
                'which_subject' : random_indices})
    else:
        return({'simulated_images' : simulated_image_list,
                'simulated_segmentation_images' : simulated_segmentation_image_list,
                'simulated_transforms' : simulated_transforms,
                'which_subject' : random_indices})
