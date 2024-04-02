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

from .registration import registration


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
        fixed = utils.slice_image(image, axis=idim - 1, idx=0) * 0
        for k in range(nTimePoints):
            temp = utils.slice_image(image, axis=idim - 1, idx=k)
            fixed = fixed + utils.iMath(temp,"Normalize") * wt
    if mask is None:
        mask = utils.get_mask(fixed)
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
                    fixed, temp, type_of_transform=type_of_transform, mask=useMask,
                    outprefix=outprefixloc, **kwargs
                )
            else:
                myreg = registration(
                    fixed, temp, type_of_transform=type_of_transform, mask=useMask, **kwargs
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
