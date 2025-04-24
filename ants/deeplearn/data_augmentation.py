__all__ = ["data_augmentation"]

import ants
import numpy as np
import random

def data_augmentation(input_image_list,
                      segmentation_image_list=None,
                      pointset_list=None,
                      number_of_simulations=10,
                      reference_image=None,
                      transform_type='affineAndDeformation',
                      noise_model='additivegaussian',
                      noise_parameters=(0.0, 0.05),
                      sd_simulated_bias_field=1.0,
                      sd_histogram_warping=0.05,
                      sd_affine=0.05,
                      sd_deformation=0.2,
                      output_numpy_file_prefix=None,
                      verbose=False
                     ):

    """
    Randomly transform image data.

    Given an input image list (possibly multi-modal) and an optional corresponding
    segmentation image list, this function will perform data augmentation with
    the following augmentation possibilities:

    * spatial transformations
    * added image noise
    * simulated bias field
    * histogram warping

    Arguments
    ---------

    input_image_list : list of lists of ANTsImages
        List of lists of input images to warp.  The internal list sets contain one
        or more images (per subject) which are assumed to be mutually aligned.  The
        outer list contains multiple subject lists which are randomly sampled to
        produce output image list.

    segmentation_image_list : list of ANTsImages
        List of segmentation images corresponding to the input image list (optional).

    pointset_list: list of pointsets
        Numpy arrays corresponding to the input image list (optional).  If using this
        option, the transform_type must be invertible.

    number_of_simulations : integer
        Number of simulated output image sets.  Default = 10.

    reference_image : ANTsImage
        Defines the spatial domain for all output images.  If one is not specified,
        we used the first image in the input image list.

    transform_type : string
        One of the following options: "translation", "rigid", "scaleShear", "affine",
        "deformation", "affineAndDeformation".

    noise_model : string
        'additivegaussian', 'saltandpepper', 'shot', and 'speckle'. Alternatively, one
        can specify a tuple or list of one or more of the options and one is selected
        at random with reasonable, randomized parameters.  Note that the "speckle" model
        takes much longer than the others.

    noise_parameters : tuple or array or float
        'additivegaussian': (mean, standardDeviation)
        'saltandpepper': (probability, saltValue, pepperValue)
        'shot': scale
        'speckle': standardDeviation
        Note that the standard deviation, scale, and probability values are *max* values
        and are randomly selected in the range [0, noise_parameter].  Also, the "mean",
        "saltValue" and "pepperValue" are assumed to be in the intensity normalized range
        of [0, 1].

    sd_simulated_bias_field : float
        Characterize the standard deviation of the amplitude.

    sd_histogram_warping : float
        Determines the strength of the bias field.

    sd_affine : float
        Determines the amount of affine transformation.

    sd_deformation : float
        Determines the amount of deformable transformation.

    output_numpy_file_prefix : string
        Filename of output numpy array containing all the simulated images and segmentations.

    Returns
    -------
    list of lists of transformed images and/or outputs to a numpy array.

    Example
    -------
    >>> image1_list = list()
    >>> image1_list.append(ants.image_read(ants.get_ants_data("r16")))
    >>> image2_list = list()
    >>> image2_list.append(ants.image_read(ants.get_ants_data("r64")))
    >>> segmentation1 = ants.threshold_image(image1_list[0], "Otsu", 3)
    >>> segmentation2 = ants.threshold_image(image2_list[0], "Otsu", 3)
    >>> input_segmentations = list()
    >>> input_segmentations.append(segmentation1)
    >>> input_segmentations.append(segmentation2)
    >>> points1 = ants.get_centroids(segmentation1)[:,0:2]
    >>> points2 = ants.get_centroids(segmentation2)[:,0:2]
    >>> input_points = list()
    >>> input_points.append(points1)
    >>> input_points.append(points2)
    >>> input_images = list()
    >>> input_images.append(image1_list)
    >>> input_images.append(image2_list)
    >>> data = data_augmentation(input_images,
                                 input_segmentations,
                                 input_points,
                                 tranform_type="scaleShear")
    """

    if reference_image is None:
        reference_image = input_image_list[0][0]

    number_of_modalities = len(input_image_list[0])

    # Set up numpy arrays if outputing to file.

    batch_X = None
    batch_Y = None
    batch_Y_points = None
    number_of_points = 0

    if pointset_list is not None:
        number_of_points = pointset_list[0].shape[0]
        batch_Y_points = np.zeros((number_of_simulations, number_of_points, reference_image.dimension))

    if output_numpy_file_prefix is not None:
        batch_X = np.zeros((number_of_simulations, *reference_image.shape, number_of_modalities))
        if segmentation_image_list is not None:
            batch_Y = np.zeros((number_of_simulations, *reference_image.shape))

    # Spatially transform input image data

    if verbose:
        print("Randomly spatially transforming the image data.")

    transform_augmentation = ants.randomly_transform_image_data(reference_image,
        input_image_list=input_image_list,
        segmentation_image_list=segmentation_image_list,
        number_of_simulations=number_of_simulations,
        transform_type=transform_type,
        sd_affine=sd_affine,
        deformation_transform_type="bspline",
        number_of_random_points=1000,
        sd_noise=sd_deformation,
        number_of_fitting_levels=4,
        mesh_size=1,
        sd_smoothing=4.0,
        input_image_interpolator='linear',
        segmentation_image_interpolator='nearestNeighbor')

    simulated_image_list = list()
    simulated_segmentation_image_list = list()
    simulated_pointset_list = list()

    for i in range(number_of_simulations):

        if verbose:
            print("Processing simulation " + str(i))

        segmentation = None
        if segmentation_image_list is not None:
            segmentation = transform_augmentation['simulated_segmentation_images'][i]
            simulated_segmentation_image_list.append(segmentation)
            if batch_Y is not None:
                if reference_image.dimension == 2:
                    batch_Y[i, :, :] = segmentation.numpy()
                else:
                    batch_Y[i, :, :, :] = segmentation.numpy()

        if pointset_list is not None:
            simulated_transform = transform_augmentation['simulated_transforms'][i]
            simulated_transform_inverse = ants.invert_ants_transform(simulated_transform)
            which_subject = transform_augmentation['which_subject'][i]
            simulated_points = np.zeros((number_of_points, reference_image.dimension))
            for j in range(number_of_points):
                simulated_points[j,:] = ants.apply_ants_transform_to_point(
                    simulated_transform_inverse, pointset_list[which_subject][j,:])
            simulated_pointset_list.append(simulated_points)
            if batch_Y_points is not None:
                batch_Y_points[i,:,:] = simulated_points


        simulated_local_image_list = list()
        for j in range(number_of_modalities):

            if verbose:
                print("    Modality " + str(j))

            image = transform_augmentation['simulated_images'][i][j]
            image_range = image.range()

            # Normalize to [0, 1] before applying augmentation

            if verbose:
                print("        Normalizing to [0, 1].")

            image = ants.iMath(image, "Normalize")

            # Noise

            if noise_model is not None:

                if not isinstance(noise_model, str):
                    noise_model = random.sample(noise_model, 1)[0]
                    # Just picked some parameters that looked reasonable
                    if noise_model == "additivegaussian":
                        noise_parameters = (random.uniform(0.0, 1.0),
                                            random.uniform(0.01, 0.25))
                    elif noise_model == "saltandpepper":
                        noise_parameters = (random.uniform(0.0, 0.25),
                                            random.uniform(0.0, 0.25),
                                            random.uniform(0.75, 1.0))
                    elif noise_model == "shot":
                        noise_parameters = (random.uniform(10, 1000),)
                    elif noise_model == "speckle":
                        noise_parameters = (random.uniform(0.1, 1),)
                    else:
                        raise ValueError("Unrecognized noise model.")

                if verbose:
                    print("        Adding noise (" + noise_model + ").")

                if noise_model.lower() == "additivegaussian":
                    parameters = (noise_parameters[0], random.uniform(0.0, noise_parameters[1]))
                    image = ants.add_noise_to_image(image,
                                                    noise_model="additivegaussian",
                                                    noise_parameters=parameters)
                elif noise_model.lower() == "saltandpepper":
                    parameters = (random.uniform(0.0, noise_parameters[0]), noise_parameters[1], noise_parameters[2])
                    image = ants.add_noise_to_image(image,
                                                    noise_model="saltandpepper",
                                                    noise_parameters=parameters)
                elif noise_model.lower() == "shot":
                    parameters = (random.uniform(0.0, noise_parameters[0]))
                    image = ants.add_noise_to_image(image,
                                                    noise_model="shot",
                                                    noise_parameters=parameters)
                elif noise_model.lower() == "speckle":
                    parameters = (random.uniform(0.0, noise_parameters[0]))
                    image = ants.add_noise_to_image(image,
                                                    noise_model="speckle",
                                                    noise_parameters=parameters)
                else:
                    raise ValueError("Unrecognized noise model.")

            # Simulated bias field

            if sd_simulated_bias_field > 0:

                if verbose:
                    print("        Adding simulated bias field.")

                log_field = ants.simulate_bias_field(image, 
                                                     number_of_points=10, 
                                                     sd_bias_field=sd_simulated_bias_field, 
                                                     number_of_fitting_levels=2, 
                                                     mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                image = image * ants.from_numpy_like(field_array, image)

            # Histogram intensity warping

            if sd_histogram_warping > 0:

                if verbose:
                    print("        Performing intensity histogram warping.")

                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(random.gauss(0, sd_histogram_warping))
                image = ants.histogram_warp_image_intensities(image,
                                                              break_points=break_points,
                                                              clamp_end_points=(False, False),
                                                              displacements=displacements)

            # Rescale to original intensity range

            if verbose:
                print("        Rescaling to original intensity range.")

            image = ants.iMath(image, "Normalize") * (image_range[1] - image_range[0]) + image_range[0]

            simulated_local_image_list.append(image)

            if batch_X is not None:
                if reference_image.dimension == 2:
                    batch_X[i, :, :, j] = image.numpy()
                else:
                    batch_X[i, :, :, :, j] = image.numpy()

        simulated_image_list.append(simulated_local_image_list)

    if batch_X is not None:
        if output_numpy_file_prefix is not None:
            if verbose:
                print("Writing images to numpy array.")
            np.save(output_numpy_file_prefix + "SimulatedImages.npy", batch_X)
    if batch_Y is not None:
        if output_numpy_file_prefix is not None:
            if verbose:
                print("Writing segmentation images to numpy array.")
            np.save(output_numpy_file_prefix + "SimulatedSegmentationImages.npy", batch_Y)
    if batch_Y_points is not None:
        if output_numpy_file_prefix is not None:
            if verbose:
                print("Writing segmentation images to numpy array.")
            np.save(output_numpy_file_prefix + "SimulatedPointsets.npy", batch_Y_points)

    if segmentation_image_list is None and pointset_list is None:
        return({'simulated_images' : simulated_image_list})
    elif segmentation_image_list is None:
        return({'simulated_images' : simulated_image_list,
                'simulated_pointset_list' : simulated_pointset_list})
    elif pointset_list is None:
        return({'simulated_images' : simulated_image_list,
                'simulated_segmentation_images' : simulated_segmentation_image_list})
    else:
        return({'simulated_images' : simulated_image_list,
                'simulated_segmentation_images' : simulated_segmentation_image_list,
                'simulated_pointset_list' : simulated_pointset_list})

