"""
ANTsPy Registration
"""
__all__ = ["registration", "motion_correction"]

import os
import numpy as np
from tempfile import mktemp
import glob
import re
import pandas as pd
import itertools

from . import apply_transforms
from . import apply_transforms_to_points
from .. import utils
from ..core import ants_image as iio
from .. import core


def registration(
    fixed,
    moving,
    type_of_transform="SyN",
    initial_transform=None,
    outprefix="",
    mask=None,
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
        transforms to prepend

    outprefix : string
        output will be named with this prefix.

    mask : ANTsImage (optional)
        mask the registration.

    grad_step : scalar
        gradient step size (not for all tx)

    flow_sigma : scalar
        smoothing for update field

    total_sigma : scalar
        smoothing for total field

    aff_metric : string
        the metric for the affine part (GC, mattes, meansquares)

    aff_sampling : scalar
        the nbins or radius parameter for the syn metric

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
        with the SyNOnly transformation.

    restrict_transformation : This option allows the user to restrict the
          optimization of the displacement field, translation, rigid or
          affine transform on a per-component basis. For example, if
          one wants to limit the deformation or rotation of 3-D volume
          to the first two dimensions, this is possible by specifying a
          weight vector of ‘(1,1,0)’ for a 3D deformation field or
          ‘(1,1,0,1,1,0)’ for a rigid transformation. Restriction
          currently only works if there are no preceding
          transformations.

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
        - "Similarity": Similarity transformation: scaling, rotation and translation.
        - "QuickRigid": Rigid transformation: Only rotation and translation.
                        May be useful for quick visualization fixes.'
        - "DenseRigid": Rigid transformation: Only rotation and translation.
                        Employs dense sampling during metric estimation.'
        - "BOLDRigid": Rigid transformation: Parameters typical for BOLD to
                        BOLD intrasubject registration'.'
        - "Affine": Affine transformation: Rigid + scaling.
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
        - "SyNOnly": Symmetric normalization: no initial transformation,
                    with mutual information as optimization metric. Assumes
                    images are aligned by an inital transformation. Can be
                    useful if you want to run an unmasked affine followed by
                    masked deformable registration.
        - "SyNCC": SyN, but with cross-correlation as the metric.
        - "SyNabp": SyN optimized for abpBrainExtraction.
        - "SyNBold": SyN, but optimized for registrations between BOLD and T1 images.
        - "SyNBoldAff": SyN, but optimized for registrations between BOLD
                        and T1 images, with additional affine step.
        - "SyNAggro": SyN, but with more aggressive registration
                        (fine-scale matching and more deformation).
                        Takes more time than SyN.
        - "TV[n]": time-varying diffeomorphism with where 'n' indicates number of
            time points in velocity field discretization.  The initial transform
            should be computed, if needed, in a separate call to ants.registration.
        - "TVMSQ": time-varying diffeomorphism with mean square metric
        - "TVMSQC": time-varying diffeomorphism with mean square metric for very large deformation
        - "antsRegistrationSyN[x]": recreation of the antsRegistrationSyN.sh script in ANTs
                                    where 'x' is one of the transforms available (e.g., 't', 'b', 's')
        - "antsRegistrationSyNQuick[x]": recreation of the antsRegistrationSyNQuick.sh script in ANTs
                                    where 'x' is one of the transforms available (e.g., 't', 'b', 's')
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
    """
    if isinstance(fixed, list) and (moving is None):
        processed_args = utils._int_antsProcessArguments(fixed)
        libfn = utils.get_lib_fn("antsRegistration")
        reg_exit = libfn(processed_args)
        if (reg_exit != 0):
            raise RuntimeError(f"Registration failed with error code {reg_exit}")
        else:
            return 0

    if not (isinstance(fixed, iio.ANTsImage) and isinstance(moving, iio.ANTsImage)):
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

    args = [fixed, moving, type_of_transform, outprefix]
    myl = 0
    myf_aff = "6x4x2x1"  # old fixed params
    mys_aff = "3x2x1x0"  # old fixed params
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
        myf_aff = aff_shrink_factors
        mys_aff = aff_smoothing_sigmas
        myiterations = aff_iterations

    if restrict_transformation is not None:
        if type(restrict_transformation) is tuple:
            restrict_transformationchar = "x".join([str(ri) for ri in restrict_transformation])

    if type(aff_shrink_factors) is tuple:
        myf_aff = "x".join([str(ri) for ri in aff_shrink_factors])
        mys_aff = "x".join([str(ri) for ri in aff_smoothing_sigmas])
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
        myf_aff = "2x1"
        mys_aff = "1x0"
        myiterations = "100x20"
        myl = 1
    if type_of_transform == "QuickRigid":
        type_of_transform = "Rigid"
        myiterations = "20x20x0x0"
    if type_of_transform == "DenseRigid":
        type_of_transform = "Rigid"
        aff_random_sampling_rate = 1.0
    if type_of_transform == "BOLDRigid":
        type_of_transform = "Rigid"
        myf_aff = "2x1"
        mys_aff = "1x0"
        myiterations = "100x20"
        myl = 1

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
    ttexists = False
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
        "BOLDRigid",
        "antsRegistrationSyN[r]",
        "antsRegistrationSyN[t]",
        "antsRegistrationSyN[a]",
        "antsRegistrationSyN[b]",
        "antsRegistrationSyN[s]",
        "antsRegistrationSyN[br]",
        "antsRegistrationSyN[sr]",
        "antsRegistrationSyN[bo]",
        "antsRegistrationSyN[so]",
        "antsRegistrationSyNQuick[r]",
        "antsRegistrationSyNQuick[t]",
        "antsRegistrationSyNQuick[a]",
        "antsRegistrationSyNQuick[b]",
        "antsRegistrationSyNQuick[s]",
        "antsRegistrationSyNQuick[br]",
        "antsRegistrationSyNQuick[sr]",
        "antsRegistrationSyNQuick[bo]",
        "antsRegistrationSyNQuick[so]",
        "antsRegistrationSyNRepro[r]",
        "antsRegistrationSyNRepro[t]",
        "antsRegistrationSyNRepro[a]",
        "antsRegistrationSyNRepro[b]",
        "antsRegistrationSyNRepro[s]",
        "antsRegistrationSyNRepro[br]",
        "antsRegistrationSyNRepro[sr]",
        "antsRegistrationSyNRepro[bo]",
        "antsRegistrationSyNRepro[so]",
        "antsRegistrationSyNQuickRepro[r]",
        "antsRegistrationSyNQuickRepro[t]",
        "antsRegistrationSyNQuickRepro[a]",
        "antsRegistrationSyNQuickRepro[b]",
        "antsRegistrationSyNQuickRepro[s]",
        "antsRegistrationSyNQuickRepro[br]",
        "antsRegistrationSyNQuickRepro[sr]",
        "antsRegistrationSyNQuickRepro[bo]",
        "antsRegistrationSyNQuickRepro[so]",
    }
    ttexists = type_of_transform in allowable_tx
    if not ttexists:
        raise ValueError("`type_of_transform` does not exist")

    initx = initial_transform
    # if isinstance(initx, ANTsTransform):
    # tempTXfilename = tempfile( fileext = '.mat' )
    # initx = invertAntsrTransform( initialTransform )
    # initx = invertAntsrTransform( initx )
    # writeAntsrTransform( initx, tempTXfilename )
    # initx = tempTXfilename
    moving = moving.clone("float")
    fixed = fixed.clone("float")
    # NOTE: this may be better for general purpose applications: TBD
#    moving = utils.iMath( moving.clone("float"), "Normalize" )
#    fixed = utils.iMath( fixed.clone("float"), "Normalize" )
    warpedfixout = moving.clone()
    warpedmovout = fixed.clone()
    f = utils.get_pointer_string(fixed)
    m = utils.get_pointer_string(moving)
    wfo = utils.get_pointer_string(warpedfixout)
    wmo = utils.get_pointer_string(warpedmovout)
    if mask is not None:
        mask_scale = mask - mask.min()
        mask_scale = mask_scale / mask_scale.max() * 255.0
        charmask = mask_scale.clone("unsigned char")
        maskopt = "[%s,NA]" % (utils.get_pointer_string(charmask))
    else:
        maskopt = None
    if initx is None:
        initx = "[%s,%s,1]" % (f, m)
    # ------------------------------------------------------------
    if type_of_transform == "SyNBold":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Rigid[0.25]",
            "-c",
            "[1200x1200x100,1e-6,5]",
            "-s",
            "2x1x0",
            "-f",
            "4x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "SyNBoldAff":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Rigid[0.25]",
            "-c",
            "[1200x1200x100,1e-6,5]",
            "-s",
            "2x1x0",
            "-f",
            "4x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[0.25]",
            "-c",
            "[200x20,1e-6,5]",
            "-s",
            "1x0",
            "-f",
            "2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % (synits),
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "ElasticSyN":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[0.25]",
            "-c",
            "2100x1200x200x0",
            "-s",
            "3x2x1x0",
            "-f",
            "4x2x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % (synits),
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "SyN" or type_of_transform == "Elastic":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[0.25]",
            "-c",
            "2100x1200x1200x0",
            "-s",
            "3x2x1x0",
            "-f",
            "4x2x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "SyNRA":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Rigid[0.25]",
            "-c",
            "2100x1200x1200x0",
            "-s",
            "3x2x1x0",
            "-f",
            "4x2x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[0.25]",
            "-c",
            "2100x1200x1200x0",
            "-s",
            "3x2x1x0",
            "-f",
            "4x2x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "SyNOnly":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if multivariate_extras is not None:
            metrics = []
            for kk in range(len(multivariate_extras)):
                metrics.append("-m")
                metricname = multivariate_extras[kk][0]
                metricfixed = utils.get_pointer_string(
                    multivariate_extras[kk][1]
                )
                metricmov = utils.get_pointer_string(
                    multivariate_extras[kk][2]
                )
                metricWeight = multivariate_extras[kk][3]
                metricSampling = multivariate_extras[kk][4]
                metricString = "%s[%s,%s,%s,%s]" % (
                    metricname,
                    metricfixed,
                    metricmov,
                    metricWeight,
                    metricSampling,
                )
                metrics.append(metricString)
            args = [
                "-d",
                str(fixed.dimension),
                "-r",
                initx,
                "-m",
                "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            ]
            args1 = [
                "-t",
                mysyn,
                "-c",
                "[%s,1e-7,8]" % synits,
                "-s",
                smoothingsigmas,
                "-f",
                shrinkfactors,
                "-u",
                "1",
                "-z",
                "1",
                "-l",
                myl,
                "-o",
                "[%s,%s,%s]" % (outprefix, wmo, wfo),
            ]
            for kk in range(len(metrics)):
                args.append(metrics[kk])
            for kk in range(len(args1)):
                args.append(args1[kk])
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "SyNAggro":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[0.25]",
            "-c",
            "2100x1200x1200x100",
            "-s",
            "3x2x1x0",
            "-f",
            "4x2x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "SyNCC":
        syn_metric = "CC"
        syn_sampling = 4
        synits = "2100x1200x1200x20"
        smoothingsigmas = "3x2x1x0"
        shrinkfactors = "4x3x2x1"
        mysyn = "SyN[0.15,3,0]"

        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Rigid[1]",
            "-c",
            "2100x1200x1200x0",
            "-s",
            "3x2x1x0",
            "-f",
            "4x4x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[1]",
            "-c",
            "1200x1200x100",
            "-s",
            "2x1x0",
            "-f",
            "4x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
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
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Translation[1]",
            "-c",
            myconvlow,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Rigid[1]",
            "-c",
            myconvlow,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Similarity[1]",
            "-c",
            myconvlow,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[1]",
            "-c",
            myconvhi,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[1]",
            "-c",
            myconvhi,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------s
    elif type_of_transform == "SyNabp":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "mattes[%s,%s,1,32,regular,0.25]" % (f, m),
            "-t",
            "Rigid[0.1]",
            "-c",
            "1000x500x250x100",
            "-s",
            "4x2x1x0",
            "-f",
            "8x4x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "mattes[%s,%s,1,32,regular,0.25]" % (f, m),
            "-t",
            "Affine[0.1]",
            "-c",
            "1000x500x250x100",
            "-s",
            "4x2x1x0",
            "-f",
            "8x4x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "CC[%s,%s,0.5,4]" % (f, m),
            "-t",
            "SyN[0.1,3,0]",
            "-c",
            "50x10x0",
            "-s",
            "2x1x0",
            "-f",
            "4x2x1",
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "SyNLessAggro":
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "Affine[0.25]",
            "-c",
            "2100x1200x1200x100",
            "-s",
            "3x2x1x0",
            "-f",
            "4x2x2x1",
            "-x",
            "[NA,NA]",
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            mysyn,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform in tvTypes:
        if grad_step is None:
            grad_step = 1.0
        nTimePoints = type_of_transform.split("[")[1].split("]")[0]
        tvtx = (
            "TimeVaryingVelocityField["
            + str(grad_step)
            + ","
            + nTimePoints
            + ","
            + str(flow_sigma)
            + ",0.0,"
            + str(total_sigma)
            + ",0]"
        )
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            tvtx,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "0",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    elif type_of_transform == "TVMSQ":
        if grad_step is None:
            grad_step = 1.0

        tvtx = "TimeVaryingVelocityField[%s, 4, 0.0,0.0, 0.5,0 ]" % str(
            grad_step
        )
        args = [
            "-d",
            str(fixed.dimension),
            # '-r', initx,
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-t",
            tvtx,
            "-c",
            "[%s,1e-7,8]" % synits,
            "-s",
            smoothingsigmas,
            "-f",
            shrinkfactors,
            "-u",
            "1",
            "-z",
            "0",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif type_of_transform == "TVMSQC":
        if grad_step is None:
            grad_step = 2.0

        tvtx = "TimeVaryingVelocityField[%s, 8, 1.0,0.0, 0.05,0 ]" % str(
            grad_step
        )
        args = [
            "-d",
            str(fixed.dimension),
            # '-r', initx,
            "-m",
            "demons[%s,%s,0.5,0]" % (f, m),
            "-m",
            "meansquares[%s,%s,1,0]" % (f, m),
            "-t",
            tvtx,
            "-c",
            "[1200x1200x100x20x0,0,5]",
            "-s",
            "8x6x4x2x1vox",
            "-f",
            "8x6x4x2x1",
            "-u",
            "1",
            "-z",
            "0",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif (
        (type_of_transform == "Rigid")
        or (type_of_transform == "Similarity")
        or (type_of_transform == "Translation")
        or (type_of_transform == "Affine")
    ):
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-m",
            "%s[%s,%s,1,%s,regular,%s]"
            % (aff_metric, f, m, aff_sampling, aff_random_sampling_rate),
            "-t",
            "%s[0.25]" % type_of_transform,
            "-c",
            myiterations,
            "-s",
            mys_aff,
            "-f",
            myf_aff,
            "-u",
            "1",
            "-z",
            "1",
            "-l",
            myl,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
        ]
        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")
    # ------------------------------------------------------------
    elif "antsRegistrationSyN" in type_of_transform:
        subtype_of_transform = "s"
        if "[" in type_of_transform and "]" in type_of_transform:
            subtype_of_transform = type_of_transform.split("[")[1].split(
                "]"
            )[0]

        do_quick = False
        if "Quick" in type_of_transform:
            do_quick = True

        do_repro = False
        if "Repro" in type_of_transform:
            do_repro = True

        if do_quick == True:
            rigid_convergence = "[1000x500x250x0,1e-6,10]"
        else:
            rigid_convergence = "[1000x500x250x100,1e-6,10]"
        rigid_shrink_factors = "8x4x2x1"
        rigid_smoothing_sigmas = "3x2x1x0vox"

        if do_quick == True:
            affine_convergence = "[1000x500x250x0,1e-6,10]"
        else:
            affine_convergence = "[1000x500x250x100,1e-6,10]"
        affine_shrink_factors = "8x4x2x1"
        affine_smoothing_sigmas = "3x2x1x0vox"

        linear_metric="MI[%s,%s,1,32,Regular,0.25]"
        if do_repro == True:
            linear_metric="GC[%s,%s,1,1,Regular,0.25]"

        if do_quick == True:
            syn_convergence = "[100x70x50x0,1e-6,10]"
            syn_metric = "MI[%s,%s,1,32]" % (f, m)
        else:
            syn_convergence = "[100x70x50x20,1e-6,10]"
            syn_metric = "CC[%s,%s,1,4]" % (f, m)
        syn_shrink_factors = "8x4x2x1"
        syn_smoothing_sigmas = "3x2x1x0vox"

        if do_quick == True and do_repro == True:
            syn_convergence = "[100x70x50x0,1e-6,10]"
            syn_metric = "CC[%s,%s,1,2]" % (f, m)

        if random_seed is None and do_repro == True:
            random_seed = str( 1 )

        tx = "Rigid"
        if subtype_of_transform == "t":
            tx = "Translation"

        rigid_stage = [
            "--transform",
            tx + "[0.1]",
            "--metric",
            linear_metric % (f, m),
            "--convergence",
            rigid_convergence,
            "--shrink-factors",
            rigid_shrink_factors,
            "--smoothing-sigmas",
            rigid_smoothing_sigmas,
        ]

        affine_stage = [
            "--transform",
            "Affine[0.1]",
            "--metric",
            linear_metric % (f, m),
            "--convergence",
            affine_convergence,
            "--shrink-factors",
            affine_shrink_factors,
            "--smoothing-sigmas",
            affine_smoothing_sigmas,
        ]

        if subtype_of_transform == "sr" or subtype_of_transform == "br":
            if do_quick == True:
                syn_convergence = "[50x0,1e-6,10]"
            else:
                syn_convergence = "[50x20,1e-6,10]"
            syn_shrink_factors = "2x1"
            syn_smoothing_sigmas = "1x0vox"

        syn_stage = [
            "--metric",
            syn_metric,
            "--convergence",
            syn_convergence,
            "--shrink-factors",
            syn_shrink_factors,
            "--smoothing-sigmas",
            syn_smoothing_sigmas,
        ]

        if (
            subtype_of_transform == "b"
            or subtype_of_transform == "br"
            or subtype_of_transform == "bo"
        ):
            syn_stage.insert(0, "BSplineSyN[0.1,26,0,3]")
            syn_stage.insert(0, "--transform")

        if (
            subtype_of_transform == "s"
            or subtype_of_transform == "sr"
            or subtype_of_transform == "so"
        ):
            syn_stage.insert(0, "SyN[0.1,3,0]")
            syn_stage.insert(0, "--transform")

        args = [
            "-d",
            str(fixed.dimension),
            "-r",
            initx,
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
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

        if maskopt is not None:
            args.append("-x")
            args.append(maskopt)
        else:
            args.append("-x")
            args.append("[NA,NA]")

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
    args.append("1")
    args.append("--write-composite-transform")
    args.append(write_composite_transform * 1)
    if verbose:
        args.append("-v")
        args.append("1")

    processed_args = utils._int_antsProcessArguments(args)
    libfn = utils.get_lib_fn("antsRegistration")
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
        fixed = utils.slice_image(image, axis=idim - 1, idx=0) * 0
        for k in range(nTimePoints):
            temp = utils.slice_image(image, axis=idim - 1, idx=k)
            fixed = fixed + utils.iMath(temp,"Normalize") * wt
    if mask is None:
        mask = utils.get_mask(fixed)
    FD = np.zeros(nTimePoints)
    motion_parameters = list()
    motion_corrected = list()
    centerOfMass = mask.get_center_of_mass()
    npts = pow(2, idim - 1)
    pointOffsets = np.zeros((npts, idim - 1))
    myrad = np.ones(idim - 1).astype(int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = core.make_image(mask, mask1vals)
    myoffsets = utils.get_neighborhood_in_mask(
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
        temp = utils.slice_image(image, axis=idim - 1, idx=k)
        temp = utils.iMath(temp, "Normalize")
        if temp.numpy().var() > 0:
            if outprefix != "":
                outprefixloc = outprefix + "_" + str.zfill( str(k), 5 ) + "_"
                myreg = registration(
                    fixed, temp, type_of_transform=type_of_transform, mask=mask,
                    outprefix=outprefixloc, **kwargs
                )
            else:
                myreg = registration(
                    fixed, temp, type_of_transform=type_of_transform, mask=mask, **kwargs
                )
            fdptsTxI = apply_transforms_to_points(
                idim - 1, fdpts, myreg["fwdtransforms"]
            )
            if k > 0 and motion_parameters[k - 1] != "NA":
                fdptsTxIminus1 = apply_transforms_to_points(
                    idim - 1, fdpts, motion_parameters[k - 1]
                )
            else:
                fdptsTxIminus1 = fdptsTxI
            # take the absolute value, then the mean across columns, then the sum
            FD[k] = (fdptsTxIminus1 - fdptsTxI).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
            mywarped = apply_transforms( fixed,
                utils.slice_image(image, axis=idim - 1, idx=k),
                myreg["fwdtransforms"] )
            motion_corrected.append(mywarped)
        else:
            motion_parameters.append("NA")
            motion_corrected.append(temp)

    if verbose:
        print("Done")
    return {
        "motion_corrected": utils.list_to_ndimage(image, motion_corrected),
        "motion_parameters": motion_parameters,
        "FD": FD,
    }
