


__all__ = ['resample_image',
           'resample_image_to_target']

import os

from ants.decorators import image_method
from ants.internal import get_lib_fn, process_arguments

import ants

@image_method
def resample_image(image, resample_params, use_voxels=False, interp_type=1):
    """
    Resample image by spacing or number of voxels with
    various interpolators. Works with multi-channel images.

    ANTsR function: `resampleImage`

    Arguments
    ---------
    image : ANTsImage
        input image

    resample_params : tuple/list
        vector of size dimension with numeric values

    use_voxels : boolean
        True means interpret resample params as voxel counts

    interp_type : integer
        one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data("r16"))
    >>> finn = ants.resample_image(fi,(50,60),True,0)
    >>> filin = ants.resample_image(fi,(1.5,1.5),False,1)
    >>> img = ants.image_read( ants.get_ants_data("r16"))
    >>> img = ants.merge_channels([img, img])
    >>> outimg = ants.resample_image(img, (128,128), True)
    """
    if image.components == 1:
        inimage = image.clone('float')
        outimage = image.clone('float')
        rsampar = 'x'.join([str(rp) for rp in resample_params])

        args = [image.dimension, inimage, outimage, rsampar, int(use_voxels), interp_type]
        processed_args = process_arguments(args)
        libfn = get_lib_fn('ResampleImage')
        libfn(processed_args)
        outimage = outimage.clone(image.pixeltype)
        return outimage
    else:
        images = ants.split_channels(image)
        new_images = []
        for image in images:
            inimage = image.clone('float')
            outimage = image.clone('float')
            rsampar = 'x'.join([str(rp) for rp in resample_params])

            args = [image.dimension, inimage, outimage, rsampar, int(use_voxels), interp_type]
            processed_args = process_arguments(args)
            libfn = get_lib_fn('ResampleImage')
            libfn(processed_args)
            outimage = outimage.clone(image.pixeltype)
            new_images.append(outimage)
        outimage = ants.merge_channels(new_images, channels_first=image.channels_first)
        return outimage

@image_method
def resample_image_to_target(image, target, interp_type='linear', imagetype=0, verbose=False, **kwargs):
    """
    Resample image by using another image as target reference.
    This function uses ants.apply_transform with an identity matrix
    to achieve proper resampling.

    ANTsR function: `resampleImageToTarget`

    Arguments
    ---------
    image : ANTsImage
        image to resample

    target : ANTsImage
        image of reference, the output will be in this space and will have the same pixel type.

    interp_type : string
        Choice of interpolator. Supports partial matching.
            linear
            nearestNeighbor
            multiLabel for label images but genericlabel is preferred
            gaussian
            bSpline
            cosineWindowedSinc
            welchWindowedSinc
            hammingWindowedSinc
            lanczosWindowedSinc
            genericLabel use this for label images

    imagetype : integer
        choose 0/1/2/3 mapping to scalar/vector/tensor/time-series

    verbose : boolean
        print command and run verbose application of transform.

    kwargs : keyword arguments
        additional arugment passed to antsApplyTransforms C code

    Returns
    -------
    ANTsImage

    Example
    -------
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
        if ants.is_image(fixed) and ants.is_image(moving):
            inpixeltype = fixed.pixeltype
            warpedmovout = moving.clone()
            f = fixed.clone('float')
            m = moving.clone('float')
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

            myargs = process_arguments(args)

            myverb = int(verbose)

            processed_args = myargs + ['-z', str(1), '-v', str(myverb), '--float', str(1), '-e', str(imagetype)]
            libfn = get_lib_fn('antsApplyTransforms')
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
        processed_args = myargs + ['-z', str(1), '--float', str(1), '-e', str(imagetype)]
        libfn = get_lib_fn('antsApplyTransforms')
        libfn(processed_args)
