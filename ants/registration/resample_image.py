
 

__all__ = ['resample_image',
           'resample_image_to_target']

import os

from ..core import ants_image as iio
from .. import utils
from .. import lib

def resample_image(img, resample_params, use_voxels=False, interp_type=1):
    if img.components == 1:
        inimg = img.clone('double')
        outimg = img.clone('double')
        rsampar = 'x'.join([str(rp) for rp in resample_params])

        args = [img.dimension, inimg, outimg, rsampar, int(use_voxels), interp_type]
        processed_args = utils._int_antsProcessArguments(args)
        lib.ResampleImage(processed_args)
        outimg = outimg.clone(img.pixeltype)
        return outimg
    else:
        raise ValueError('images with more than 1 component not currently supported')


def resample_image_to_target(image, target, interp_type='linear', imagetype=0, verbose=False, **kwargs):
    """
    Examples
    --------
    >>> import ants
    >>> fi = ants.image_read(ants.get_ants_data('r16'))
    >>> fi2mm = ants.resample_image(fi, (2,2), use_voxels=0, interp_type='linear')
    >>> resampled = ants.resample_image_to_target(fi2mm, fi, verbose=True)
    """
    fixed = target
    moving = image
    compose = None
    transformlist = 'identity'
    interpolator = interp_type

    interpolator_oldoptions = ("linear", "nearestNeighbor", "gaussian", "cosineWindowedSinc", "bSpline")
    if isinstance(interp_type, int):
        interpolator = interpolator_oldoptions[interp_type]

    accepted_interpolators = {"linear", "nearestNeighbor", "multiLabel", "gaussian",
                        "bSpline", "cosineWindowedSinc", "welchWindowedSinc",
                        "hammingWindowedSinc", "lanczosWindowedSinc", "genericLabel"}

    if interpolator not in accepted_interpolators:
        raise ValueError('interpolator not supported - see %s' % accepted_interpolators)

    args = [fixed, moving, transformlist, interpolator]

    if not isinstance(fixed, str):
        if isinstance(fixed, iio.ANTsImage) and isinstance(moving, iio.ANTsImage):
            inpixeltype = fixed.pixeltype
            warpedmovout = moving.clone()
            f = fixed
            m = moving
            if (moving.dimension == 4) and (fixed.dimension==3) and (imagetype==0):
                raise ValueError('Set imagetype 3 to transform time series images.')

            wmo = warpedmovout
            mytx = ['-t', 'identity']
            if compose is None:
                args = ['-d', fixed.dimension, '-i', m, '-o', wmo, '-r', f, '-n', interpolator] + mytx

            tfn = '%scomptx.nii.gz' % compose if compose is not None else 'NA'
            if compose is not None:
                mycompo = '[%s,1]' % tfn
                args = ['-d', fixed.dimension, '-i', m, '-o', mycompo, '-r', f, '-n', interpolator] + mytx

            myargs = utils._int_antsProcessArguments(args)

            # NO CLUE WHAT THIS DOES OR WHY IT'S NEEDED
            for jj in range(len(myargs)):
                if myargs[jj] is not None:
                    if myargs[jj] == '-':
                        myargs2 = [None]*(len(myargs)-1)
                        myargs2[:(jj-1)] = myargs[:(jj-1)]
                        myargs2[jj:(len(myargs)-1)] = myargs[(jj+1):(len(myargs))]
                        myargs = myargs2

            myverb = int(verbose)

            processed_args = myargs + ['-z', str(1), '-v', str(myverb), '--float', str(1), '-e', str(imagetype)]
            lib.antsApplyTransforms(processed_args)

            if compose is None:
                return warpedmovout.clone(inpixeltype)
            else:
                if os.path.exists(tfn):
                    return tfn
                else:
                    return None
        else:
            return 1
    else:
        processed_args = myargs + ['-z', str(1), '--float', str(1), '-e', str(imagetype)]
        lib.antsApplyTransforms(processed_args)
