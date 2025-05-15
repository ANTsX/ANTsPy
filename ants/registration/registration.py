"""
ANTsPy Registration
"""
__all__ = ["registration",
           "motion_correction",
           "label_image_registration"]

import numpy as np
from tempfile import mktemp
import glob
import re
import pandas as pd
import itertools
import os

import ants
from ants.internal import get_lib_fn, get_pointer_string, process_arguments

def registration(
    fixed,
    moving,
    type_of_transform="SyN",
    initial_transform=None,
    outprefix="",
    mask=None,
    moving_mask=None,
    mask_all_stages=False,
    grad_step=0.2,
    flow_sigma=3,
    total_sigma=0,
    aff_metric="mattes",
    aff_sampling=32,
    aff_random_sampling_rate=0.2,
    syn_metric="mattes",
    syn_sampling=32,
    reg_iterations=(40, 20, 0),
    aff_iterations=(2100, 1200, 1200, 10),
    aff_shrink_factors=(6, 4, 2, 1),
    aff_smoothing_sigmas=(3, 2, 1, 0),
    write_composite_transform=False,
    random_seed=None,
    verbose=False,
    multivariate_extras=None,
    restrict_transformation=None,
    smoothing_in_mm=False,
    singleprecision=True,
    use_legacy_histogram_matching=False,
    **kwargs
):
    """
    Register a pair of images either through the full or simplified
    interface to the ANTs registration method.

    ANTsR function: `antsRegistration`

    Arguments
    ---------
    fixed : ANTsImage
        fixed image to which we register the moving image.

    moving : ANTsImage
        moving image to be mapped to fixed space.

    type_of_transform : string
        A linear or non-linear registration type. Mutual information metric by default.
        See Notes below for more.

    initial_transform : list of strings (optional)
        transforms to prepend. If None, a translation is computed to align the image centers of mass, unless the type of
        transform is deformable-only (time-varying diffeomorphisms, SyNOnly, or antsRegistrationSyN*[so|bo]).
        To force initialization with an identity transform, set this to 'Identity'.

    outprefix : string
        output will be named with this prefix.

    mask : ANTsImage (optional)
        Registration metric mask in the fixed image space.

    moving_mask : ANTsImage (optional)
        Registration metric mask in the moving image space.

    mask_all_stages : boolean
        If true, apply metric mask(s) to all registration stages, instead of just the final stage.

    grad_step : scalar
        gradient step size (not for all tx)

    flow_sigma : scalar
        smoothing for update field
        At each iteration, the similarity metric and gradient is calculated.
        That gradient field is also called the update field and is smoothed
        before composing with the total field (i.e., the estimate of the total
        transform at that iteration). This total field can also be smoothed
        after each iteration.

    total_sigma : scalar
        smoothing for total field

    aff_metric : string
        the metric for the affine part (GC, mattes, meansquares)

    aff_sampling : scalar
        number of bins for the mutual information metric

    aff_random_sampling_rate : scalar
        the fraction of points used to estimate the metric. this can impact
        speed but also reproducibility and/or accuracy.

    syn_metric : string
        the metric for the syn part (CC, mattes, meansquares, demons)

    syn_sampling : scalar
        the nbins or radius parameter for the syn metric

    reg_iterations : list/tuple of integers
        vector of iterations for syn. we will set the smoothing and multi-resolution parameters based on the length of this vector.

    aff_iterations : list/tuple of integers
        vector of iterations for low-dimensional (translation, rigid, affine) registration.

    aff_shrink_factors : list/tuple of integers
        vector of multi-resolution shrink factors for low-dimensional (translation, rigid, affine) registration.

    aff_smoothing_sigmas : list/tuple of integers
        vector of multi-resolution smoothing factors for low-dimensional (translation, rigid, affine) registration.

    random_seed : integer
        random seed to improve reproducibility. note that the number of ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS should be 1 if you want perfect reproducibility.

    write_composite_transform : boolean
        Boolean specifying whether or not the composite transform (and its inverse, if it exists) should be written to an hdf5 composite file. This is false by default so that only the transform for each stage is written to file.

    verbose : boolean
        request verbose output (useful for debugging)

    multivariate_extras : additional metrics for multi-metric registration
        list of additional images and metrics which will
        trigger the use of multiple metrics in the registration
        process in the deformable stage. Each multivariate metric needs 5
        entries: name of metric, fixed, moving, weight,
        samplingParam. the list of lists should be of the form ( (
        "nameOfMetric2", img, img, weight, metricParam ) ). Another
        example would be  ( ( "MeanSquares", f2, m2, 0.5, 0
          ), ( "CC", f2, m2, 0.5, 2 ) ) .  This is only compatible
        with the SyNOnly or antsRegistrationSyN* transformations.

    restrict_transformation : This option allows the user to restrict the
          optimization of the displacement field, translation, rigid or
          affine transform on a per-component basis. For example, if
          one wants to limit the deformation or rotation of 3-D volume
          to the first two dimensions, this is possible by specifying a
          weight vector of ‘(1,1,0)’ for a 3D deformation field or
          ‘(1,1,0,1,1,0)’ for a rigid transformation. Restriction
          currently only works if there are no preceding
          transformations.

    smoothing_in_mm : boolean ; currently only impacts low dimensional registration

    singleprecision : boolean
        if True, use float32 for computations. This is useful for reducing memory
        usage for large datasets, at the cost of precision.

    use_legacy_histogram_matching : boolean
        if True, use the original histogram matching in ANTs. This is not recommended, but is available for backwards
        compatibilty with earlier versions, where it was always turned on. The default is False. A better implementation of
        histogram matching is available in the ants.histogram_match_image2 function.

    kwargs : keyword args
        extra arguments

    Returns
    -------
    dict containing follow key/value pairs:
        `warpedmovout`: Moving image warped to space of fixed image.
        `warpedfixout`: Fixed image warped to space of moving image.
        `fwdtransforms`: Transforms to move from moving to fixed image.
        `invtransforms`: Transforms to move from fixed to moving image.

    Notes
    -----
    type_of_transform can be one of:
        - "Translation": Translation transformation.
        - "Rigid": Rigid transformation: Only rotation and translation.
        - "Similarity": Similarity transformation: uniform scaling, rotation and translation.
        - "QuickRigid": Rigid transformation: Only rotation and translation.
                        May be useful for quick visualization fixes.'
        - "DenseRigid": Rigid transformation: Only rotation and translation.
                        Employs dense sampling during metric estimation.'
        - "BOLDRigid": Rigid transformation: Parameters typical for BOLD to
                        BOLD intrasubject registration'.'
        - "Affine": Affine transformation: Rigid + scaling + shear (12 parameters).
        - "AffineFast": Fast version of Affine.
        - "BOLDAffine": Affine transformation: Parameters typical for BOLD to
                        BOLD intrasubject registration'.'
        - "TRSAA": translation, rigid, similarity, affine (twice). please set
                    regIterations if using this option. this would be used in
                    cases where you want a really high quality affine mapping
                    (perhaps with mask).
        - "Elastic": Elastic deformation: Affine + deformable.
        - "ElasticSyN": Symmetric normalization: Affine + deformable
                        transformation, with mutual information as optimization
                        metric and elastic regularization.
        - "SyN": Symmetric normalization: Affine + deformable transformation,
                    with mutual information as optimization metric.
        - "SyNRA": Symmetric normalization: Rigid + Affine + deformable
                    transformation, with mutual information as optimization metric.
        - "SyNOnly": Symmetric normalization with no rigid or affine stages.
                    Uses mutual information as optimization metric.
        - "SyNCC": SyN, but with cross-correlation as the metric.
        - "SyNabp": SyN optimized for abpBrainExtraction.
        - "SyNBold": SyN, but optimized for registrations between BOLD and T1 images.
        - "SyNBoldAff": SyN, but optimized for registrations between BOLD
                        and T1 images, with additional affine step.
        - "SyNAggro": SyN, but with more aggressive registration
                        (fine-scale matching and more deformation).
                        Takes more time than SyN.
        - "SyNLessAggro": Does exactly the same thing as "SyNAggro".
        - "TV[n]": time-varying diffeomorphism with where 'n' indicates number of
            time points in velocity field discretization.  The initial transform
            should be computed, if needed, in a separate call to ants.registration.
        - "TVMSQ": time-varying diffeomorphism with mean square metric
        - "TVMSQC": time-varying diffeomorphism with mean square metric for very large deformation
        - "antsRegistrationSyN[x]": recreation of the antsRegistrationSyN.sh script in ANTs
                                    where 'x' is one of the transforms available:
                                         t: translation (1 stage)
                                         r: rigid (1 stage)
                                         a: rigid + affine (2 stages)
                                         s: rigid + affine + deformable syn (3 stages)
                                         sr: rigid + deformable syn (2 stages)
                                         so: deformable syn only (1 stage)
                                         b: rigid + affine + deformable b-spline syn (3 stages)
                                         br: rigid + deformable b-spline syn (2 stages)
                                         bo: deformable b-spline syn only (1 stage)
        - "antsRegistrationSyNQuick[x]": recreation of the antsRegistrationSyNQuick.sh script in ANTs.
                                         x options as above.
        - "antsRegistrationSyNRepro[x]": reproducible registration.  x options as above.
        - "antsRegistrationSyNQuickRepro[x]": quick reproducible registration.  x options as above.

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16'))
    >>> mi = ants.image_read(ants.get_ants_data('r64'))
    >>> fi = ants.resample_image(fi, (60,60), 1, 0)
    >>> mi = ants.resample_image(mi, (60,60), 1, 0)
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN' )
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'antsRegistrationSyN[t]' )
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'antsRegistrationSyN[b]' )
    >>> mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'antsRegistrationSyN[s]' )
    """
    if isinstance(fixed, list) and (moving is None):
        processed_args = process_arguments(fixed)
        libfn = get_lib_fn("antsRegistration")
        reg_exit = libfn(processed_args)
        if (reg_exit != 0):
            raise RuntimeError(f"Registration failed with error code {reg_exit}")
        else:
            return 0

    if not (ants.is_image(fixed) and ants.is_image(moving)):
        raise ValueError("Fixed and moving images must be ANTsImage objects")

    if type_of_transform == "":
        type_of_transform = "SyN"

    if isinstance(type_of_transform, (tuple, list)) and (len(type_of_transform) == 1):
        type_of_transform = type_of_transform[0]

    if (outprefix == "") or len(outprefix) == 0:
        outprefix = mktemp()

    if np.sum(np.isnan(fixed.numpy())) > 0:
        raise ValueError("fixed image has NaNs - replace these")
    if np.sum(np.isnan(moving.numpy())) > 0:
        raise ValueError("moving image has NaNs - replace these")

    if fixed.dimension != moving.dimension:
        raise ValueError("Fixed and moving image dimensions are not the same.")
    # ----------------------------

    myiterations = aff_iterations
    args = [fixed, moving, type_of_transform, outprefix]
    shrinkfactors_affine = "6x4x2x1"  # old fixed params
    smoothingsigmas_affine = "3x2x1x0"  # old fixed params
    if (
        type(aff_shrink_factors) is int
        or type(aff_smoothing_sigmas) is int
        or type(aff_iterations) is int
    ):
        if type(aff_smoothing_sigmas) is not int:
            raise ValueError("aff_smoothing_sigmas should be a single integer.")
        if type(aff_iterations) is not int:
            raise ValueError("aff_iterations should be a single integer.")
        if type(aff_shrink_factors) is not int:
            raise ValueError("aff_shrink_factors should be a single integer.")
        shrinkfactors_affine = aff_shrink_factors
        smoothingsigmas_affine = aff_smoothing_sigmas
        myiterations = aff_iterations

    if restrict_transformation is not None:
        if type(restrict_transformation) is tuple:
            restrict_transformationchar = "x".join([str(ri) for ri in restrict_transformation])

    if type(aff_shrink_factors) is tuple:
        shrinkfactors_affine = "x".join([str(ri) for ri in aff_shrink_factors])
        smoothingsigmas_affine = "x".join([str(ri) for ri in aff_smoothing_sigmas])
        myiterations = "x".join([str(ri) for ri in aff_iterations])
        if len(aff_iterations) != len(aff_smoothing_sigmas):
            raise ValueError(
                "aff_iterations length should equal aff_smoothing_sigmas length."
            )
        if len(aff_iterations) != len(aff_shrink_factors):
            raise ValueError(
                "aff_iterations length should equal aff_shrink_factors length."
            )
        if len(aff_shrink_factors) != len(aff_smoothing_sigmas):
            raise ValueError(
                "aff_shrink_factors length should equal aff_smoothing_sigmas length."
            )

    if type_of_transform == "AffineFast":
        type_of_transform = "Affine"
        myiterations = "2100x1200x0x0"
    if type_of_transform == "BOLDAffine":
        type_of_transform = "Affine"
        shrinkfactors_affine = "2x1"
        smoothingsigmas_affine = "1x0"
        myiterations = "100x20"
    if type_of_transform == "QuickRigid":
        type_of_transform = "Rigid"
        myiterations = "20x20x0x0"
    if type_of_transform == "DenseRigid":
        type_of_transform = "Rigid"
        aff_random_sampling_rate = 1.0
    if type_of_transform == "BOLDRigid":
        type_of_transform = "Rigid"
        shrinkfactors_affine = "2x1"
        smoothingsigmas_affine = "1x0"
        myiterations = "100x20"

    if smoothing_in_mm:
        smoothingsigmas_affine = smoothingsigmas_affine + 'mm'

    mysyn = "SyN[%f,%f,%f]" % (grad_step, flow_sigma, total_sigma)
    if type_of_transform == "Elastic":
        mysyn = "GaussianDisplacementField[%f,%f,%f]" % (grad_step, flow_sigma, total_sigma)
    itlen = len(reg_iterations)  # NEED TO CHECK THIS
    if itlen == 0:
        smoothingsigmas = 0
        shrinkfactors = 1
        synits = reg_iterations
    else:
        smoothingsigmas = np.arange(0, itlen)[::-1].astype(
            "float32"
        )  # NEED TO CHECK THIS
        shrinkfactors = 2 ** smoothingsigmas
        shrinkfactors = shrinkfactors.astype("int")
        smoothingsigmas = "x".join([str(ss)[0] for ss in smoothingsigmas])
        shrinkfactors = "x".join([str(ss) for ss in shrinkfactors])
        synits = "x".join([str(ri) for ri in reg_iterations])

    inpixeltype = fixed.pixeltype
    output_pixel_type = 'float' if singleprecision else 'double'

    tvTypes = [
        "TV[1]",
        "TV[2]",
        "TV[3]",
        "TV[4]",
        "TV[5]",
        "TV[6]",
        "TV[7]",
        "TV[8]",
    ]
    allowable_tx = {
        "SyNBold",
        "SyNBoldAff",
        "ElasticSyN",
        "Elastic",
        "SyN",
        "SyNRA",
        "SyNOnly",
        "SyNAggro",
        "SyNCC",
        "TRSAA",
        "SyNabp",
        "SyNLessAggro",
        "TV[1]",
        "TV[2]",
        "TV[3]",
        "TV[4]",
        "TV[5]",
        "TV[6]",
        "TV[7]",
        "TV[8]",
        "TVMSQ",
        "TVMSQC",
        "Rigid",
        "Similarity",
        "Translation",
        "Affine",
        "AffineFast",
        "BOLDAffine",
        "QuickRigid",
        "DenseRigid",
        "BOLDRigid"
    }
    deformable_only_transforms = [
        "SyNOnly",
        "antsRegistrationSyN[so]",
        "antsRegistrationSyNQuick[so]",
        "antsRegistrationSyNRepro[so]",
        "antsRegistrationSyNQuickRepro[so]",
        "antsRegistrationSyN[bo]",
        "antsRegistrationSyNQuick[bo]",
        "antsRegistrationSyNRepro[bo]",
        "antsRegistrationSyNQuickRepro[bo]",
        "TVMSQ",
        "TVMSQC"
    ] + tvTypes
    transform_type_exists = type_of_transform in allowable_tx

    # Perform checking of antsRegistrationSyN transforms later
    if not "antsRegistrationSyN" in type_of_transform and not transform_type_exists:
        raise ValueError(f'{type_of_transform} does not exist')

    if isinstance(initial_transform, str):
        initial_transform = [initial_transform]
    # if isinstance(initx, ANTsTransform):
    # tempTXfilename = tempfile( fileext = '.mat' )
    # initx = invertAntsrTransform( initialTransform )
    # initx = invertAntsrTransform( initx )
    # writeAntsrTransform( initx, tempTXfilename )
    # initial_transform = tempTXfilename
    moving = moving.clone(output_pixel_type)
    fixed = fixed.clone(output_pixel_type)
    # NOTE: this may be better for general purpose applications: TBD
#    moving = ants.iMath( moving.clone("float"), "Normalize" )
#    fixed = ants.iMath( fixed.clone("float"), "Normalize" )
    warpedfixout = moving.clone()
    warpedmovout = fixed.clone()
    fixed_str = get_pointer_string(fixed)
    moving_str = get_pointer_string(moving)
    warpedfixout_str = get_pointer_string(warpedfixout)
    warpedmovout_str = get_pointer_string(warpedmovout)
    if mask is not None:
        mask_binary = mask != 0
        fixed_mask_str = get_pointer_string(mask_binary)
    else:
        fixed_mask_str = "NA"

    if moving_mask is not None:
        moving_mask_binary = moving_mask != 0
        moving_mask_str = get_pointer_string(moving_mask_binary)
    else:
        moving_mask_str = "NA"

    maskopt = "[%s,%s]" % (fixed_mask_str, moving_mask_str)

    if mask_all_stages:
        earlymaskopt = maskopt
    else:
        earlymaskopt = "[NA,NA]"

    if initial_transform is None:
        if type_of_transform in deformable_only_transforms:
            initial_transform = ["Identity"]
        else:
            initial_transform = ["[%s,%s,1]" % (fixed_str, moving_str)]

    # ------------------------------------------------------------
    if type_of_transform == "SyNBold":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Rigid[0.25]",
            "--convergence", "[1200x1200x100,1e-6,5]",
            "--smoothing-sigmas", "2x1x0",
            "--shrink-factors", "4x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", mysyn,
            "--convergence", "[%s,1e-7,8]" % synits,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "SyNBoldAff":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Rigid[0.25]",
            "--convergence", "[1200x1200x100,1e-6,5]",
            "--smoothing-sigmas", "2x1x0",
            "--shrink-factors", "4x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[0.25]",
            "--convergence", "[200x20,1e-6,5]",
            "--smoothing-sigmas", "1x0",
            "--shrink-factors", "2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", mysyn,
            "--convergence", "[%s,1e-7,8]" % (synits),
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "ElasticSyN":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[0.25]",
            "--convergence", "2100x1200x200x0",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x2x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", mysyn,
            "--convergence", "[%s,1e-7,8]" % (synits),
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "SyN" or type_of_transform == "Elastic":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[0.25]",
            "--convergence", "2100x1200x1200x0",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x2x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", mysyn,
            "--convergence", "[%s,1e-7,8]" % synits,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "SyNRA":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Rigid[0.25]",
            "--convergence", "2100x1200x1200x0",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x2x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[0.25]",
            "--convergence", "2100x1200x1200x0",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x2x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", mysyn,
            "--convergence", "[%s,1e-7,8]" % synits,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "SyNOnly":
        if multivariate_extras is None:
            args = [
                "--dimensionality", str(fixed.dimension),
                "-r"] + initial_transform + [
                "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
                "--transform", mysyn,
                "--convergence", "[%s,1e-7,8]" % synits,
                "--smoothing-sigmas", smoothingsigmas,
                "--shrink-factors", shrinkfactors,
                "-u", str(int(use_legacy_histogram_matching)),
                "-z", "1",
                "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            ]
        else:
            metrics = []
            for mve_idx in range(len(multivariate_extras)):
                metrics.append("--metric")
                metric_name = multivariate_extras[mve_idx][0]
                metric_fixed_str = get_pointer_string(
                    multivariate_extras[mve_idx][1]
                )
                metric_moving_str = get_pointer_string(
                    multivariate_extras[mve_idx][2]
                )
                metric_weight = multivariate_extras[mve_idx][3]
                metric_sampling = multivariate_extras[mve_idx][4]
                metric_full_string = "%s[%s,%s,%s,%s]" % (
                    metric_name,
                    metric_fixed_str,
                    metric_moving_str,
                    metric_weight,
                    metric_sampling,
                )
                metrics.append(metric_full_string)
            args_pre = [
                "--dimensionality", str(fixed.dimension),
                "-r"] + initial_transform + [
                "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            ]
            args_post = [
                "--transform", mysyn,
                "--convergence", "[%s,1e-7,8]" % synits,
                "--smoothing-sigmas", smoothingsigmas,
                "--shrink-factors", shrinkfactors,
                "-u", str(int(use_legacy_histogram_matching)),
                "-z", "1",
                "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            ]
            args = args_pre + metrics + args_post
        args.append("-x")
        args.append(maskopt)
    # ------------------------------------------------------------
    elif type_of_transform == "SyNAggro":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[0.25]",
            "--convergence", "2100x1200x1200x100",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x2x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", mysyn,
            "--convergence", "[%s,1e-7,8]" % synits,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "SyNCC":
        # syn_metric = "CC"
        # syn_sampling = 4
        # synits = "2100x1200x1200x20"
        # smoothingsigmas = "3x2x1x0"
        # shrinkfactors = "4x3x2x1"
        # mysyn = "SyN[0.15,3,0]"
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Rigid[1]",
            "--convergence", "2100x1200x1200x0",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x4x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[1]",
            "--convergence", "1200x1200x100",
            "--smoothing-sigmas", "2x1x0",
            "--shrink-factors", "4x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % ("CC", fixed_str, moving_str, 4),
            "--transform", "SyN[0.15,3,0]",
            "--convergence", "[2100x1200x1200x20,1e-7,8]",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x3x2x1",
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "TRSAA":
        itlen = len(reg_iterations)
        itlenlow = round(itlen / 2 + 0.0001)
        dlen = itlen - itlenlow
        _myconvlow = [2000] * itlenlow + [0] * dlen
        myconvlow = "x".join([str(mc) for mc in _myconvlow])
        myconvhi = "x".join([str(r) for r in reg_iterations])
        myconvhi = "[%s,1.e-7,10]" % myconvhi
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Translation[1]",
            "--convergence", myconvlow,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Rigid[1]",
            "--convergence", myconvlow,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Similarity[1]",
            "--convergence", myconvlow,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[1]",
            "--convergence", myconvhi,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[1]",
            "--convergence", myconvhi,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------s
    elif type_of_transform == "SyNabp":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "mattes[%s,%s,1,32,regular,0.25]" % (fixed_str, moving_str),
            "--transform", "Rigid[0.1]",
            "--convergence", "1000x500x250x100",
            "--smoothing-sigmas", "4x2x1x0",
            "--shrink-factors", "8x4x2x1",
            "-x", earlymaskopt,
            "--metric", "mattes[%s,%s,1,32,regular,0.25]" % (fixed_str, moving_str),
            "--transform", "Affine[0.1]",
            "--convergence", "1000x500x250x100",
            "--smoothing-sigmas", "4x2x1x0",
            "--shrink-factors", "8x4x2x1",
            "-x", earlymaskopt,
            "--metric", "CC[%s,%s,0.5,4]" % (fixed_str, moving_str),
            "--transform", "SyN[0.1,3,0]",
            "--convergence", "50x10x0",
            "--smoothing-sigmas", "2x1x0",
            "--shrink-factors", "4x2x1",
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "SyNLessAggro":
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "Affine[0.25]",
            "--convergence", "2100x1200x1200x100",
            "--smoothing-sigmas", "3x2x1x0",
            "--shrink-factors", "4x2x2x1",
            "-x", earlymaskopt,
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", mysyn,
            "--convergence", "[%s,1e-7,8]" % synits,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform in tvTypes:
        if grad_step is None:
            grad_step = 1.0
        nTimePoints = type_of_transform.split("[")[1].split("]")[0]
        tvtx = (
            "TimeVaryingVelocityField["
            + ",".join([
                str(grad_step),
                nTimePoints,
                str(flow_sigma),
                "0.0",
                str(total_sigma),
                "0"])
            +"]"
        )
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", tvtx,
            "--convergence", "[%s,1e-7,8]" % synits,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "0",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    elif type_of_transform == "TVMSQ":
        if grad_step is None:
            grad_step = 1.0
        tvtx = "TimeVaryingVelocityField[%s, 4, 0.0,0.0, 0.5,0 ]" % str(
            grad_step
        )
        args = [
            "--dimensionality",
            str(fixed.dimension),
            '-r'
        ] + initial_transform + [
            "--metric",
            "%s[%s,%s,1,%s]" % (syn_metric, fixed_str, moving_str, syn_sampling),
            "--transform", tvtx,
            "--convergence", "[%s,1e-7,8]" % synits,
            "--smoothing-sigmas", smoothingsigmas,
            "--shrink-factors", shrinkfactors,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "0",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform == "TVMSQC":
        if grad_step is None:
            grad_step = 2.0

        tvtx = "TimeVaryingVelocityField[%s, 8, 1.0,0.0, 0.05,0 ]" % str(
            grad_step
        )
        args = [
            "--dimensionality", str(fixed.dimension),
            '-r'] + initial_transform + [
            "--metric", "demons[%s,%s,0.5,0]" % (fixed_str, moving_str),
            "--metric", "meansquares[%s,%s,1,0]" % (fixed_str, moving_str),
            "--transform", tvtx,
            "--convergence", "[1200x1200x100x20x0,0,5]",
            "--smoothing-sigmas", "8x6x4x2x1vox",
            "--shrink-factors", "8x6x4x2x1",
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "0",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif type_of_transform in ("Rigid", "Similarity", "Translation", "Affine"):
        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--metric", "%s[%s,%s,1,%s,regular,%s]" % (aff_metric, fixed_str, moving_str, aff_sampling, aff_random_sampling_rate),
            "--transform", "%s[0.25]" % type_of_transform,
            "--convergence", myiterations,
            "--smoothing-sigmas", smoothingsigmas_affine,
            "--shrink-factors", shrinkfactors_affine,
            "-u", str(int(use_legacy_histogram_matching)),
            "-z", "1",
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
            "-x", maskopt
        ]
    # ------------------------------------------------------------
    elif "antsRegistrationSyN" in type_of_transform:
        do_quick = ("Quick" in type_of_transform)
        do_repro = ("Repro" in type_of_transform)

        subtype_of_transform = "s"
        spline_distance = 26
        metric_parameter = 32 if do_quick else 2
        linear_metric = "GC[%s,%s,1,1,Regular,0.25]" if do_repro else "MI[%s,%s,1,32,Regular,0.25]"

        rigid_shrink_factors = "8x4x2x1"
        rigid_smoothing_sigmas = "3x2x1x0vox"

        affine_shrink_factors = "8x4x2x1"
        affine_smoothing_sigmas = "3x2x1x0vox"
        
        linear_gradient_step = 0.1
        syn_gradient_step = 0.2

        if "[" in type_of_transform and "]" in type_of_transform:
            subtype_of_transform = type_of_transform.split("[")[1].split(
                "]"
            )[0]
            if "," in subtype_of_transform:
                subtype_of_transform_args = subtype_of_transform.split(",")
                subtype_of_transform = subtype_of_transform_args[0]
                if not ( subtype_of_transform in ["b", "br", "bo", "s", "sr", "so"]):
                    raise ValueError("Extra parameters are only valid for 's' or 'b' SyN transforms.")
                metric_parameter = subtype_of_transform_args[1]
                if len(subtype_of_transform_args) > 2:
                    spline_distance = subtype_of_transform_args[2]

        if do_quick:
            rigid_convergence = "[1000x500x250x0,1e-6,10]"
            affine_convergence = "[1000x500x250x0,1e-6,10]"
            syn_convergence = "[100x70x50x0,1e-6,10]"
            if do_repro:
                metric_parameter = 2
                syn_metric = "CC[%s,%s,1,%s]" % (fixed_str, moving_str, metric_parameter)
            else:
                metric_parameter = 32
                syn_metric = "MI[%s,%s,1,%s]" % (fixed_str, moving_str, metric_parameter)
        else:
            rigid_convergence = "[1000x500x250x100,1e-6,10]"
            affine_convergence = "[1000x500x250x100,1e-6,10]"
            syn_convergence = "[100x70x50x20,1e-6,10]"
            metric_parameter = 2
            syn_metric = "CC[%s,%s,1,%s]" % (fixed_str, moving_str, metric_parameter)
        syn_shrink_factors = "8x4x2x1"
        syn_smoothing_sigmas = "3x2x1x0vox"
        if subtype_of_transform in ("sr", "br"):
            if do_quick:
                syn_convergence = "[50x0,1e-6,10]"
            else:
                syn_convergence = "[50x20,1e-6,10]"
            syn_shrink_factors = "2x1"
            syn_smoothing_sigmas = "1x0vox"

        if random_seed is None and do_repro:
            random_seed = "1"

        rigidtx = "Translation" if subtype_of_transform == "t" else "Rigid"

        rigid_stage = [
            "--transform", rigidtx + "[" + str(linear_gradient_step) + "]",
            "--metric", linear_metric % (fixed_str, moving_str),
            "--convergence", rigid_convergence,
            "--shrink-factors", rigid_shrink_factors,
            "--smoothing-sigmas", rigid_smoothing_sigmas,
        ]
        affine_stage = [
            "--transform", "Affine[" + str(linear_gradient_step) + "]",
            "--metric", linear_metric % (fixed_str, moving_str),
            "--convergence", affine_convergence,
            "--shrink-factors", affine_shrink_factors,
            "--smoothing-sigmas", affine_smoothing_sigmas,
        ]
        syn_stage = [
            "--metric", syn_metric,
        ]
        if multivariate_extras is not None:
            for mve_idx in range(len(multivariate_extras)):
                syn_stage.append("--metric")
                metric_name = multivariate_extras[mve_idx][0]
                metric_fixed_str = get_pointer_string(
                    multivariate_extras[mve_idx][1]
                )
                metric_moving_str = get_pointer_string(
                    multivariate_extras[mve_idx][2]
                )
                metric_weight = multivariate_extras[mve_idx][3]
                metric_sampling = multivariate_extras[mve_idx][4]
                metric_full_string = "%s[%s,%s,%s,%s]" % (
                    metric_name,
                    metric_fixed_str,
                    metric_moving_str,
                    metric_weight,
                    metric_sampling,
                )
                syn_stage.append(metric_full_string)

        syn_stage.append("--convergence")
        syn_stage.append(syn_convergence)
        syn_stage.append("--shrink-factors")
        syn_stage.append(syn_shrink_factors)
        syn_stage.append("--smoothing-sigmas")
        syn_stage.append(syn_smoothing_sigmas)

        if subtype_of_transform in ("b", "br", "bo"):
            syn_stage.insert(0, "BSplineSyN[" + str(syn_gradient_step) + "," + str(spline_distance) + ",0,3]")
            syn_stage.insert(0, "--transform")

        if subtype_of_transform in ("s", "sr", "so"):
            syn_stage.insert(0, "SyN[" + str(syn_gradient_step) + ",3,0]")
            syn_stage.insert(0, "--transform")

        args = [
            "--dimensionality", str(fixed.dimension),
            "-r"] + initial_transform + [
            "--output", "[%s,%s,%s]" % (outprefix, warpedmovout_str, warpedfixout_str),
        ]

        if subtype_of_transform == "r" or subtype_of_transform == "t":
            args.append(rigid_stage)
        if subtype_of_transform == "a":
            args.append(rigid_stage)
            args.append(affine_stage)
        if subtype_of_transform == "b" or subtype_of_transform == "s":
            args.append(rigid_stage)
            args.append(affine_stage)
            args.append(syn_stage)
        if subtype_of_transform == "br" or subtype_of_transform == "sr":
            args.append(rigid_stage)
            args.append(syn_stage)
        if subtype_of_transform == "bo" or subtype_of_transform == "so":
            args.append(syn_stage)

        args.append("-x")
        args.append(maskopt)

        args = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, 1) if isinstance(x, str) else x
                for x in args
            )
        )

    # ------------------------------------------------------------

    if random_seed is not None:
        args.append("--random-seed")
        args.append(random_seed)

    if restrict_transformation is not None:
        args.append("-g")
        args.append(restrict_transformationchar)

    args.append("--float")
    args.append(str(int(singleprecision)))
    args.append("--write-composite-transform")
    args.append(write_composite_transform * 1)
    if verbose:
        args.append("-v")
        args.append("1")

    processed_args = process_arguments(args)
    libfn = get_lib_fn("antsRegistration")
    if verbose:
        print("antsRegistration " + ' '.join(processed_args))
    reg_exit = libfn(processed_args)
    if (reg_exit != 0):
        raise RuntimeError(f"Registration failed with error code {reg_exit}")
    afffns = glob.glob(outprefix + "*" + "[0-9]GenericAffine.mat")
    fwarpfns = glob.glob(outprefix + "*" + "[0-9]Warp.nii.gz")
    iwarpfns = glob.glob(outprefix + "*" + "[0-9]InverseWarp.nii.gz")
    vfieldfns = glob.glob(outprefix + "*" + "[0-9]VelocityField.nii.gz")
    # print(afffns, fwarpfns, iwarpfns)
    if len(afffns) == 0:
        afffns = ""
    if len(fwarpfns) == 0:
        fwarpfns = ""
    if len(iwarpfns) == 0:
        iwarpfns = ""
    if len(vfieldfns) == 0:
        vfieldfns = ""

    alltx = sorted(
        set(glob.glob(outprefix + "*" + "[0-9]*"))
        - set(glob.glob(outprefix + "*VelocityField*"))
    )
    findinv = np.where(
        [re.search("[0-9]InverseWarp.nii.gz", ff) for ff in alltx]
    )[0]
    findfwd = np.where([re.search("[0-9]Warp.nii.gz", ff) for ff in alltx])[
        0
    ]
    if len(findinv) > 0:
        fwdtransforms = list(
            reversed(
                [ff for idx, ff in enumerate(alltx) if idx != findinv[0]]
            )
        )
        invtransforms = [
            ff for idx, ff in enumerate(alltx) if idx != findfwd[0]
        ]
    else:
        fwdtransforms = list(reversed(alltx))
        invtransforms = alltx

    if write_composite_transform:
        fwdtransforms = outprefix + "Composite.h5"
        invtransforms = outprefix + "InverseComposite.h5"

    if not vfieldfns:
        return {
            "warpedmovout": warpedmovout.clone(inpixeltype),
            "warpedfixout": warpedfixout.clone(inpixeltype),
            "fwdtransforms": fwdtransforms,
            "invtransforms": invtransforms,
        }
    else:
        return {
            "warpedmovout": warpedmovout.clone(inpixeltype),
            "warpedfixout": warpedfixout.clone(inpixeltype),
            "fwdtransforms": fwdtransforms,
            "invtransforms": invtransforms,
            "velocityfield": vfieldfns,
        }

def motion_correction(
    image,
    fixed=None,
    type_of_transform="BOLDRigid",
    mask=None,
    fdOffset=50,
    outprefix="",
    verbose=False,
    **kwargs
):
    """
    Correct time-series data for motion.

    ANTsR function: `antsrMotionCalculation`

    Arguments
    ---------
        image: antsImage, usually ND where D=4.

        fixed: Fixed image to register all timepoints to.  If not provided,
            mean image is used.

        type_of_transform : string
            A linear or non-linear registration type. Mutual information metric and rigid transformation by default.
            See ants registration for details.

        mask: mask for image (ND-1).  If not provided, estimated from data.
            2023-02-05: a performance change - previously, we estimated a mask
            when None is provided and would pass this to the registration.  this
            impairs performance if the mask estimate is bad.  in such a case, we
            prefer no mask at all.  As such, we no longer pass the mask to the
            registration when None is provided.

        fdOffset: offset value to use in framewise displacement calculation

        outprefix : string
            output will be named with this prefix plus a numeric extension.

        verbose: boolean

        kwargs: keyword args
            extra arguments - these extra arguments will control the details of registration that is performed. see ants registration for more.

    Returns
    -------
    dict containing follow key/value pairs:
        `motion_corrected`: Moving image warped to space of fixed image.
        `motion_parameters`: transforms for each image in the time series.
        `FD`: Framewise displacement generalized for arbitrary transformations.

    Notes
    -----
    Control extra arguments via kwargs. see ants.registration for details.

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('ch2'))
    >>> mytx = ants.motion_correction( fi )
    """
    idim = image.dimension
    ishape = image.shape
    nTimePoints = ishape[idim - 1]
    if fixed is None:
        wt = 1.0 / nTimePoints
        fixed = ants.slice_image(image, axis=idim - 1, idx=0) * 0
        for k in range(nTimePoints):
            temp = ants.slice_image(image, axis=idim - 1, idx=k)
            fixed = fixed + ants.iMath(temp,"Normalize") * wt
    if mask is None:
        mask = ants.get_mask(fixed)
        useMask=None
    else:
        useMask=mask
    FD = np.zeros(nTimePoints)
    motion_parameters = list()
    motion_corrected = list()
    centerOfMass = mask.get_center_of_mass()
    npts = pow(2, idim - 1)
    pointOffsets = np.zeros((npts, idim - 1))
    myrad = np.ones(idim - 1).astype(int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(
        mask1, mask1, radius=myrad, spatial_info=True
    )["offsets"]

    mycols = list("xy")
    if idim - 1 == 3:
        mycols = list("xyz")
    useinds = list()
    for k in range(myoffsets.shape[0]):
        if abs(myoffsets[k, :]).sum() == (idim - 2):
            useinds.append(k)
        myoffsets[k, :] = myoffsets[k, :] * fdOffset / 2.0 + centerOfMass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)
    if verbose:
        print("Progress:")
    counter = 0
    for k in range(nTimePoints):
        mycount = round(k / nTimePoints * 100)
        if verbose and mycount == counter:
            counter = counter + 10
            print(mycount, end="%.", flush=True)
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "Normalize")
        if temp.numpy().var() > 0:
            if outprefix != "":
                outprefixloc = outprefix + "_" + str.zfill( str(k), 5 ) + "_"
                myreg = registration(
                    fixed, temp, type_of_transform=type_of_transform, mask=useMask,
                    outprefix=outprefixloc, **kwargs
                )
            else:
                myreg = registration(
                    fixed, temp, type_of_transform=type_of_transform, mask=useMask, **kwargs
                )
            fdptsTxI = ants.apply_transforms_to_points(
                idim - 1, fdpts, myreg["fwdtransforms"]
            )
            if k > 0 and motion_parameters[k - 1] != "NA":
                fdptsTxIminus1 = ants.apply_transforms_to_points(
                    idim - 1, fdpts, motion_parameters[k - 1]
                )
            else:
                fdptsTxIminus1 = fdptsTxI
            # take the absolute value, then the mean across columns, then the sum
            FD[k] = (fdptsTxIminus1 - fdptsTxI).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
            mywarped = ants.apply_transforms( fixed,
                ants.slice_image(image, axis=idim - 1, idx=k),
                myreg["fwdtransforms"] )
            motion_corrected.append(mywarped)
        else:
            motion_parameters.append("NA")
            motion_corrected.append(temp)

    if verbose:
        print("Done")
    return {
        "motion_corrected": ants.list_to_ndimage(image, motion_corrected),
        "motion_parameters": motion_parameters,
        "FD": FD,
    }

def label_image_registration(fixed_label_images,
                             moving_label_images,
                             fixed_intensity_images=None,
                             moving_intensity_images=None,
                             fixed_mask=None,
                             moving_mask=None,
                             initial_transforms='affine',
                             type_of_deformable_transform='antsRegistrationSyNQuick[so]',
                             label_image_weighting=1.0,
                             output_prefix='',
                             random_seed=None,
                             verbose=False):

    """
    Perform pairwise registration using fixed and moving sets of label
    images (and, optionally, sets of corresponding intensity images).

    Arguments
    ---------
    fixed_label_images : single or list of ANTsImage
        A single (or set of) fixed label image(s).

    moving_label_images : single or list of ANTsImage
        A single (or set of) moving label image(s).

    fixed_intensity_images : single or list of ANTsImage
        Optional---a single (or set of) fixed intensity image(s).

    moving_intensity_images : single or list of ANTsImage
        Optional---a single (or set of) moving intensity image(s).

    fixed_mask : ANTsImage
        Defines region for similarity metric calculation in the space
        of the fixed image.

    moving_mask : ANTsImage
        Defines region for similarity metric calculation in the space
        of the moving image.

    initial_transforms : string or list of files
        If specified, there are two options:  2) Use label images with 
        the centers of mass to a calculate linear transform of type 
        'identity', 'rigid', 'similarity', or 'affine'.  2) Specify a
        list of transform files, e.g., the output of ants.registration().

    type_of_deformable_transform : string
        Only works with deformable-only transforms, specifically the family
        of antsRegistrationSyN*[so] or antsRegistrationSyN*[bo] transforms.
        See 'type_of_transform' in ants.registration.  Additionally, one can
        use a list to pass a more tailored deformably-only transform
        optimization using SyN or BSplineSyN transforms.  The order of
        parameters in the list would be 1) transform specification, i.e.
        "SyN" or "BSplineSyN", 2) gradient (real), 3) intensity metric (string),
        4) intensity metric parameter (real), 5) convergence iterations per level
        (tuple) 6) smoothing factors per level (tuple), 7) shrink factors per level
        (tuple).  An example would type_of_deformable_transform = ["SyN", 0.2, "CC",
        4, (100,50,10), (2,1,0), (4,2,1)].

    label_image_weighting : float or list of floats
        Relative weighting for the label images.

    output_prefix : string
        Define the output prefix for the filenames of the output transform
        files.

    random_seed : integer
        Definition for deformable registration.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Set of transforms definining the mapping to/from the fixed image domain
    to the moving image domain.

    Example
    -------
    >>> import ants
    >>>
    >>> r16 = ants.image_read(ants.get_ants_data('r16'))
    >>> r16_seg1 = ants.threshold_image(r16, "Kmeans", 3) - 1
    >>> r16_seg2 = ants.threshold_image(r16, "Kmeans", 5) - 1
    >>> r64 = ants.image_read(ants.get_ants_data('r64'))
    >>> r64_seg1 = ants.threshold_image(r64, "Kmeans", 3) - 1
    >>> r64_seg2 = ants.threshold_image(r64, "Kmeans", 5) - 1
    >>> reg = ants.label_image_registration([r16_seg1, r16_seg2],
                                            [r64_seg1, r64_seg2],
                                            fixed_intensity_images=r16,
                                            moving_intensity_images=r64,
                                            initial_transforms='affine',
                                            type_of_deformable_transform='antsRegistrationSyNQuick[bo]',
                                            label_image_weighting=[1.0, 2.0],
                                            verbose=True)
    """

    # Perform validation check on the input

    if isinstance(fixed_label_images, ants.ANTsImage):
        fixed_label_images = [ants.image_clone(fixed_label_images)]
    if isinstance(moving_label_images, ants.ANTsImage):
        moving_label_images = [ants.image_clone(moving_label_images)]

    if len(fixed_label_images) != len(moving_label_images):
        raise ValueError("The number of fixed and moving label images do not match.")

    if fixed_intensity_images is not None or moving_intensity_images is not None:
        if isinstance(fixed_intensity_images, ants.ANTsImage):
            fixed_intensity_images = [ants.image_clone(fixed_intensity_images)]
        if isinstance(moving_intensity_images, ants.ANTsImage):
            moving_intensity_images = [ants.image_clone(moving_intensity_images)]
        if len(fixed_intensity_images) != len(moving_intensity_images):
            raise ValueError("The number of fixed and moving intensity images do not match.")

    label_image_weights = list()
    if isinstance(label_image_weighting, (int, float)):
        label_image_weights = [label_image_weighting] * len(fixed_label_images)
    else:
        label_image_weights = tuple(label_image_weighting)
        if len(fixed_label_images) != len(label_image_weights):
            raise ValueError("The length of label_image_weights must" +
                             "match the number of label image pairs.")

    image_dimension = fixed_label_images[0].dimension

    if output_prefix == "" or output_prefix is None or len(output_prefix) == 0:
        output_prefix = mktemp()

    do_deformable = True
    if type_of_deformable_transform is None or len(type_of_deformable_transform) == 0:
       do_deformable = False

    common_label_ids = list()
    total_number_of_labels = 0
    for i in range(len(fixed_label_images)):
        fixed_label_geoms = ants.label_geometry_measures(fixed_label_images[i])
        fixed_label_ids = np.array(fixed_label_geoms['Label'])
        moving_label_geoms = ants.label_geometry_measures(moving_label_images[i])
        moving_label_ids = np.array(moving_label_geoms['Label'])
        common_label_ids.append(np.intersect1d(moving_label_ids, fixed_label_ids))
        total_number_of_labels += len(common_label_ids[i])
        if verbose:
            print("Common label ids for image pair ", str(i), ": ", common_label_ids[i])
        if len(common_label_ids[i]) == 0:
            raise ValueError("No common labels for image pair " + str(i))

    deformable_multivariate_extras = list()

    if verbose:
        print("Total number of labels: " + str(total_number_of_labels))

    initial_xfrm_files = list()

    ##############################
    #
    #    Initial linear transform
    #
    ##############################

    if isinstance(initial_transforms, str) and initial_transforms in ['rigid', 'similarity', 'affine']:

        if verbose:
            print("\n\nComputing linear transform.\n")

        if total_number_of_labels < 3:
            raise ValueError("  Number of labels must be >= 3.")

        fixed_centers_of_mass = np.zeros((total_number_of_labels, image_dimension))
        moving_centers_of_mass = np.zeros((total_number_of_labels, image_dimension))
        count = 0
        for i in range(len(common_label_ids)):
            for j in range(len(common_label_ids[i])):
                label = common_label_ids[i][j]
                if verbose:
                    print("  Finding centers of mass for image pair " + str(i) + ", label " + str(label))
                fixed_single_label_image = ants.threshold_image(fixed_label_images[i], label, label, 1, 0)
                fixed_centers_of_mass[count, :] = ants.get_center_of_mass(fixed_single_label_image)
                moving_single_label_image = ants.threshold_image(moving_label_images[i], label, label, 1, 0)
                moving_centers_of_mass[count, :] = ants.get_center_of_mass(moving_single_label_image)
                count += 1
                if do_deformable:
                    deformable_multivariate_extras.append(["MSQ", fixed_single_label_image,
                                                           moving_single_label_image,
                                                           label_image_weights[i], 0])

        linear_xfrm = ants.fit_transform_to_paired_points(moving_centers_of_mass,
                                                          fixed_centers_of_mass,
                                                          transform_type=initial_transforms,
                                                          verbose=verbose)
        
        if do_deformable:
            linear_xfrm_file = output_prefix + "LandmarkBasedLinear" + initial_transforms + ".mat"
        else:
            linear_xfrm_file = output_prefix + "0GenericAffine.mat"

        ants.write_transform(linear_xfrm, linear_xfrm_file)
        initial_xfrm_files.append(linear_xfrm_file)

    elif initial_transforms is not None or initial_transforms == 'identity':
        
        if do_deformable:
            for i in range(len(common_label_ids)):
                for j in range(len(common_label_ids[i])):
                    label = common_label_ids[i][j]
                    fixed_single_label_image = ants.threshold_image(fixed_label_images[i], label, label, 1, 0)
                    moving_single_label_image = ants.threshold_image(moving_label_images[i], label, label, 1, 0)
                    deformable_multivariate_extras.append(["MSQ", fixed_single_label_image,
                                                        moving_single_label_image,
                                                        label_image_weights[i], 0])

        if initial_transforms != 'identity':
            if not isinstance(initial_transforms, list):
                initial_transforms = [initial_transforms]
            for i in range(len(initial_transforms)):
                if not os.path.exists(initial_transforms[i]):
                    raise ValueError(initial_transforms[i] + " does not exist.")
                else:
                    initial_xfrm_files.append(initial_transforms[i])
            
    ##############################
    #
    #    Deformable transform
    #
    ##############################

    if do_deformable:

        if verbose:
            print("\n\nComputing deformable transform using images.\n")

        intensity_metric = "CC"
        intensity_metric_parameter = 2
        syn_shrink_factors = "8x4x2x1"
        syn_smoothing_sigmas = "3x2x1x0vox"
        syn_convergence = "[100x70x50x20,1e-6,10]"
        spline_distance = 26
        gradient_step = 0.2
        syn_transform = "SyN"

        syn_stage = list()

        if isinstance(type_of_deformable_transform, list):

            if (len(type_of_deformable_transform) != 7 or
                not isinstance(type_of_deformable_transform[0], str) or
                not isinstance(type_of_deformable_transform[1], float) or
                not isinstance(type_of_deformable_transform[2], str) or
                not isinstance(type_of_deformable_transform[3], int) or
                not isinstance(type_of_deformable_transform[4], tuple) or
                not isinstance(type_of_deformable_transform[5], tuple) or
                not isinstance(type_of_deformable_transform[6], tuple)):
                raise ValueError("Incorrect specification for type_of_deformable_transform.  See help menu.")

            syn_transform = type_of_deformable_transform[0]
            gradient_step = type_of_deformable_transform[1]
            intensity_metric = type_of_deformable_transform[2]
            intensity_metric_parameter = type_of_deformable_transform[3]

            t = type_of_deformable_transform[4]
            tstr = ''.join(map(lambda x: str(x) + 'x', t[:len(t)-1])) + str(t[len(t)-1])
            syn_convergence = "[" + tstr + ",1e-6,10]"

            t = type_of_deformable_transform[5]
            tstr = ''.join(map(lambda x: str(x) + 'x', t[:len(t)-1])) + str(t[len(t)-1])
            syn_smoothing_sigmas = tstr + "vox"

            t = type_of_deformable_transform[6]
            syn_shrink_factors = ''.join(map(lambda x: str(x) + 'x', t[:len(t)-1])) + str(t[len(t)-1])

        else:

            do_quick = False
            if "Quick" in type_of_deformable_transform:
                do_quick = True
            elif "Repro" in type_of_deformable_transform:
                random_seed = str(1)

            if "[" in type_of_deformable_transform and "]" in type_of_deformable_transform:
                subtype_of_deformable_transform = type_of_deformable_transform.split("[")[1].split("]")[0]
                if not ('bo' in subtype_of_deformable_transform or 'so' in subtype_of_deformable_transform):
                    raise ValueError("Only 'so' or 'bo' transforms are available.")
                else:
                    if 'bo' in subtype_of_deformable_transform:
                        syn_transform = "BSplineSyN"
                if "," in subtype_of_deformable_transform:
                    subtype_of_deformable_transform_args = subtype_of_deformable_transform.split(",")
                    subtype_of_deformable_transform = subtype_of_deformable_transform_args[0]
                    intensity_metric_parameter = subtype_of_deformable_transform_args[1]
                    if len(subtype_of_deformable_transform_args) > 2:
                        spline_distance = subtype_of_deformable_transform_args[2]

            if do_quick:
                intensity_metric = "MI"
                intensity_metric_parameter = 32
                syn_convergence = "[100x70x50x0,1e-6,10]"

        if fixed_intensity_images is not None and len(fixed_intensity_images) > 0:
            for i in range(len(fixed_intensity_images)):
                syn_stage.append("--metric")
                metric_string = "%s[%s,%s,%s,%s]" % (
                    intensity_metric,
                    get_pointer_string(fixed_intensity_images[i]),
                    get_pointer_string(moving_intensity_images[i]),
                    1.0, intensity_metric_parameter)
                syn_stage.append(metric_string)

        for kk in range(len(deformable_multivariate_extras)):
            syn_stage.append("--metric")
            metricString = "%s[%s,%s,%s,%s]" % (
                "MSQ",
                get_pointer_string(deformable_multivariate_extras[kk][1]),
                get_pointer_string(deformable_multivariate_extras[kk][2]),
                deformable_multivariate_extras[kk][3], 0.0)
            syn_stage.append(metricString)

        syn_stage.append("--convergence")
        syn_stage.append(syn_convergence)
        syn_stage.append("--shrink-factors")
        syn_stage.append(syn_shrink_factors)
        syn_stage.append("--smoothing-sigmas")
        syn_stage.append(syn_smoothing_sigmas)

        if syn_transform == "SyN":
            syn_stage.insert(0, "SyN[" + str(gradient_step) + ",3,0]")
        else:
            syn_stage.insert(0, "BSplineSyN[" + str(gradient_step) + "," + str(spline_distance) + ",0,3]")
        syn_stage.insert(0, "--transform")

        args = ["--dimensionality", str(image_dimension),
                "--output", output_prefix]

        if len(initial_xfrm_files) > 0:
          for i in range(len(initial_xfrm_files)):
            initial_args = ["-r", initial_xfrm_files[i]]
            args.append(initial_args)

        args.append(syn_stage)

        fixed_mask_string = 'NA'
        if fixed_mask is not None:
            fixed_mask_binary = fixed_mask != 0
            fixed_mask_string = get_pointer_string(fixed_mask_binary)

        moving_mask_string = 'NA'
        if moving_mask is not None:
            moving_mask_binary = moving_mask != 0
            moving_mask_string = get_pointer_string(moving_mask_binary)

        mask_option = "[%s,%s]" % (fixed_mask_string, moving_mask_string)

        args.append("-x")
        args.append(mask_option)

        args = list(itertools.chain.from_iterable(
                    itertools.repeat(x, 1)
                    if isinstance(x, str)
                    else x for x in args))

        args.append("--float")
        args.append("1")

        if random_seed is not None:
            args.append("--random-seed")
            args.append(random_seed)

        if verbose:
            args.append("-v")
            args.append("1")

        processed_args = process_arguments(args)
        if verbose:
            print("antsRegistration " + ' '.join(processed_args))

        libfn = get_lib_fn("antsRegistration")
        deformable_registration_exit_error = libfn(processed_args)

        if deformable_registration_exit_error != 0:
            raise RuntimeError(f"Registration failed with error code {deformable_registration_exit_error}")

    all_xfrms = sorted(set(glob.glob(output_prefix + "*" + "[0-9]*")))

    find_inverse_warps_idx = np.where([re.search("[0-9]InverseWarp.nii.gz", ff) for ff in all_xfrms])[0]
    find_forward_warps_idx = np.where([re.search("[0-9]Warp.nii.gz", ff) for ff in all_xfrms])[0]
    find_affines_idx = np.where([re.search("[0-9]GenericAffine.mat", ff) for ff in all_xfrms])[0]

    fwdtransforms = list()
    invtransforms = list()

    if len(find_forward_warps_idx) > 0:
        fwdtransforms.append(all_xfrms[find_forward_warps_idx[0]])
    if len(find_affines_idx) > 0:    
        fwdtransforms.append(all_xfrms[find_affines_idx[0]])
        invtransforms.append(all_xfrms[find_affines_idx[0]])
    if len(find_inverse_warps_idx) > 0:
        invtransforms.append(all_xfrms[find_inverse_warps_idx[0]])

    if verbose:
        print("\n\nResulting transforms")
        print("  fwdtransforms: ", fwdtransforms)
        print("  invtransforms: ", invtransforms)

    return {
        "fwdtransforms": fwdtransforms,
        "invtransforms": invtransforms,
    }


