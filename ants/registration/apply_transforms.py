

__all__ = ['apply_transforms','apply_transforms_to_points']

import os

from .. import core
from ..core import ants_image as iio
from .. import utils

def apply_transforms(fixed, moving, transformlist,
                     interpolator='linear', imagetype=0,
                     whichtoinvert=None, compose=None,
                     defaultvalue=0, verbose=False, **kwargs):
    """
    Apply a transform list to map an image from one domain to another.
    In image registration, one computes mappings between (usually) pairs
    of images. These transforms are often a sequence of increasingly
    complex maps, e.g. from translation, to rigid, to affine to deformation.
    The list of such transforms is passed to this function to interpolate one
    image domain into the next image domain, as below. The order matters
    strongly and the user is advised to familiarize with the standards
    established in examples.

    ANTsR function: `antsApplyTransforms`

    Arguments
    ---------
    fixed : ANTsImage
        fixed image defining domain into which the moving image is transformed.

    moving : AntsImage
        moving image to be mapped to fixed space.

    transformlist : list of strings
        list of transforms generated by ants.registration where each transform is a filename.

    interpolator : string
        Choice of interpolator. Supports partial matching.
            linear
            nearestNeighbor
            multiLabel for label images (deprecated, prefer genericLabel)
            gaussian
            bSpline
            cosineWindowedSinc
            welchWindowedSinc
            hammingWindowedSinc
            lanczosWindowedSinc
            genericLabel use this for label images

    imagetype : integer
        choose 0/1/2/3 mapping to scalar/vector/tensor/time-series

    whichtoinvert : list of booleans (optional)
        Must be same length as transformlist.
        whichtoinvert[i] is True if transformlist[i] is a matrix,
        and the matrix should be inverted. If transformlist[i] is a
        warp field, whichtoinvert[i] must be False.
        If the transform list is a matrix followed by a warp field,
        whichtoinvert defaults to (True,False). Otherwise it defaults
        to [False]*len(transformlist)).

    compose : string (optional)
        if it is a string pointing to a valid file location,
        this will force the function to return a composite transformation filename.

    defaultvalue : scalar
        Default voxel value for mappings outside the image domain.

    verbose : boolean
        print command and run verbose application of transform.

    kwargs : keyword arguments
        extra parameters

    Returns
    -------
    ANTsImage or string (transformation filename)

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
                if '.mat' in transformlist[i]:
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
            if compose:
                tfn = '%scomptx.nii.gz' % compose if not compose.endswith('.h5') else compose
            else:
                tfn = 'NA'
            if compose is not None:
                mycompo = '[%s,1]' % tfn
                args = ['-d', fixed.dimension,
                        '-i', m,
                        '-o', mycompo,
                        '-r', f,
                        '-n', interpolator]
                args = args + mytx

            myargs = utils._int_antsProcessArguments(args)

            myverb = int(verbose)
            if verbose:
                print(myargs)

            processed_args = myargs + ['-z', str(1), '-v', str(myverb), '--float', str(1), '-e', str(imagetype), '-f', str(defaultvalue)]
            libfn = utils.get_lib_fn('antsApplyTransforms')
            libfn(processed_args)

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
        args = args + ['-z', 1, '--float', 1, '-e', imagetype, '-f', defaultvalue]
        processed_args = utils._int_antsProcessArguments(args)
        libfn = utils.get_lib_fn('antsApplyTransforms')
        libfn(processed_args)






def apply_transforms_to_points( dim, points, transformlist,
                     whichtoinvert=None, verbose=False ):
    """
     Apply a transform list to map a pointset from one domain to
     another. In registration, one computes mappings between pairs of
     domains. These transforms are often a sequence of increasingly
     complex maps, e.g. from translation, to rigid, to affine to
     deformation.  The list of such transforms is passed to this
     function to interpolate one image domain into the next image
     domain, as below.  The order matters strongly and the user is
     advised to familiarize with the standards established in examples.
     Importantly, point mapping goes the opposite direction of image
     mapping, for both reasons of convention and engineering.

    ANTsR function: `antsApplyTransformsToPoints`

    Arguments
    ---------
    dim: integer
         dimensionality of the transformation.

    points: data frame
          moving point set with n-points in rows of at least dim
          columns - we maintain extra information in additional
          columns. this should be a data frame with columns names x, y, z, t.

    transformlist : list of strings
        list of transforms generated by ants.registration where each transform is a filename.

    whichtoinvert : list of booleans (optional)
        Must be same length as transformlist.
        whichtoinvert[i] is True if transformlist[i] is a matrix,
        and the matrix should be inverted. If transformlist[i] is a
        warp field, whichtoinvert[i] must be False.
        If the transform list is a matrix followed by a warp field,
        whichtoinvert defaults to (True,False). Otherwise it defaults
        to [False]*len(transformlist)).

    verbose : boolean

    Returns
    -------
    data frame of transformed points

    Example
    -------
    >>> import ants
    >>> fixed = ants.image_read( ants.get_ants_data('r16') )
    >>> moving = ants.image_read( ants.get_ants_data('r27') )
    >>> reg = ants.registration( fixed, moving, 'Affine' )
    >>> d = {'x': [128, 127], 'y': [101, 111]}
    >>> pts = pd.DataFrame(data=d)
    >>> ptsw = ants.apply_transforms_to_points( 2, pts, reg['fwdtransforms'])
    """

    if not isinstance(transformlist, (tuple, list)) and (transformlist is not None):
        transformlist = [transformlist]

    args = [dim, points, transformlist, whichtoinvert]

    for tl_path in transformlist:
        if not os.path.exists(tl_path):
            raise Exception('Transform %s does not exist' % tl_path)

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
        if '.mat' in transformlist[i]:
            ismat = True
        if whichtoinvert[i] and (not ismat):
            raise ValueError('Cannot invert transform %i (%s) because it is not a matrix' % (i, transformlist[i]))
        if whichtoinvert[i]:
            mytx = mytx + ['-t', '[%s,1]' % (transformlist[i])]
        else:
            mytx = mytx + ['-t', transformlist[i]]
    if dim == 2:
        pointsSub = points[['x','y']]
    if dim == 3:
        pointsSub = points[['x','y','z']]
    if dim == 4:
        pointsSub = points[['x','y','z','t']]
    pointImage = core.make_image( pointsSub.shape, pointsSub.values.flatten())
    pointsOut = pointImage.clone()
    args = ['-d', dim,
            '-i', pointImage,
            '-o', pointsOut ]
    args = args + mytx
    myargs = utils._int_antsProcessArguments(args)

    myverb = int(verbose)
    if verbose:
        print(myargs)

    processed_args = myargs + [ '-f', str(1), '--precision', str(0)]
    libfn = utils.get_lib_fn('antsApplyTransformsToPoints')
    libfn(processed_args)
    mynp = pointsOut.numpy()
    pointsOutDF = points.copy()
    pointsOutDF['x'] = mynp[:,0]
    if dim >= 2:
        pointsOutDF['y'] = mynp[:,1]
    if dim >= 3:
        pointsOutDF['z'] = mynp[:,2]
    if dim >= 4:
        pointsOutDF['t'] = mynp[:,3]
    return pointsOutDF
