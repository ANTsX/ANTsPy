

__all__ = ['apply_transforms']

import os

from ..core import ants_image as iio
from .. import lib
from .. import utils


def apply_transforms(fixed, moving, transformlist, 
                     interpolator='linear', imagetype=0, 
                     whichtoinvert=None, compose=None, verbose=False):
    """
    Arguments
    ---------
    interpolator : string
        Possibilities: ("linear", "nearestNeighbor", "multiLabel", "gaussian",
                        "bSpline", "cosineWindowedSinc", "welchWindowedSinc",
                        "hammingWindowedSinc", "lanczosWindowedSinc", "genericLabel")

    Example
    -------
    >>> import ants
    >>> fixed = ants.image_read( ants.get_ants_data('r16') )
    >>> moving = ants.image_read( ants.get_ants_data('r64') )
    >>> fixed = ants.resample_image(fixed, (64,64), 1, 0)
    >>> moving = ants.resample_image(moving, (64,64), 1, 0)
    >>> mytx = ants.registration(fixed=fixed , moving=moving ,
                                 type_of_transform = 'SyN' )
    >>> mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving,
                                               transformlist=mytx['fwdtransforms'] )
    """

    if not isinstance(transformlist, (tuple, list)) and (transformlist is not None):
        transformlist = [transformlist]

    accepted_interpolators = {"linear", "nearestNeighbor", "multiLabel", "gaussian",
                        "bSpline", "cosineWindowedSinc", "welchWindowedSinc",
                        "hammingWindowedSinc", "lanczosWindowedSinc", "genericLabel"}

    if interpolator not in accepted_interpolators:
        raise ValueError('interpolator not supported - see %s' % accepted_interpolators)

    args = [fixed, moving, transformlist, interpolator]

    if not isinstance(fixed, str):
        if isinstance(fixed, iio.ANTsImage) and isinstance(moving, iio.ANTsImage):
            for tl_path in transformlist:
                if not os.path.exists(tl_path):
                    raise Exception('Transform %s does not exist' % tl_path)
            
            inpixeltype = fixed.pixeltype
            fixed = fixed.clone('float')
            moving = moving.clone('float')
            warpedmovout = moving.clone()
            f = fixed
            m = moving
            if (moving.dimension == 4) and (fixed.dimension == 3) and (imagetype == 0):
                raise Exception('Set imagetype 3 to transform time series images.')

            wmo = warpedmovout
            mytx = []
            if whichtoinvert is None or (isinstance(whichtoinvert, (tuple,list)) and (sum([w is not None for w in whichtoinvert])==0)):
                if (len(transformlist) == 2) and ('.mat' in transformlist[0]) and ('.mat' not in transformlist[1]):
                    whichtoinvert = (True, False)
                else:
                    whichtoinvert = tuple([False]*len(transformlist))

            if len(whichtoinvert) != len(transformlist):
                raise ValueError('Transform list and inversion list must be the same length')

            for i in range(len(transformlist)):
                ismat = False
                if '.mat' in transformlist[0]:
                    ismat = True
                if whichtoinvert[i] and (not ismat):
                    raise ValueError('Cannot invert transform %i (%s) because it is not a matrix' % (i, transformlist[i]))
                if whichtoinvert[i]:
                    mytx = mytx + ['-t', '[%s,1]' % (transformlist[i])]
                else:
                    mytx = mytx + ['-t', transformlist[i]]

            if compose is None:
                args = ['-d', fixed.dimension,
                        '-i', m,
                        '-o', wmo,
                        '-r', f,
                        '-n', interpolator]
                args = args + mytx
            tfn = '%scomptx.nii.gz' % compose if compose is not None else 'NA'
            if compose is not None:
                mycompo = '[%s,1]' % tfn
                args = ['-d', fixed.dimension, 
                        '-i', m, 
                        '-o', mycompo,
                        '-r', f,
                        '-n', interpolator]
                args = args + mytx

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
            if verbose:
                print(myargs)

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
        args = args + ['-z', 1, '--float', 1, '-e', imagetype]
        processed_args = utils._int_antsProcessArguments(args)
        lib.antsApplyTransforms(processed_args)



