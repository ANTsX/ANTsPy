__all__ = ["build_template"]

import numpy as np
import os
import shutil
from tempfile import mktemp

import ants

def build_template(
    initial_template=None,
    image_list=None,
    iterations=3,
    gradient_step=0.2,
    blending_weight=0.75,
    weights=None,
    useNoRigid=True,
    output_dir=None,
    **kwargs
):
    """
    Estimate an optimal template from an input image_list

    ANTsR function: N/A

    Arguments
    ---------
    initial_template : ANTsImage
        initialization for the template building

    image_list : ANTsImages
        images from which to estimate template

    iterations : integer
        number of template building iterations

    gradient_step : scalar
        for shape update gradient

    blending_weight : scalar
        weight for image blending

    weights : vector
        weight for each input image

    useNoRigid : boolean
        equivalent of -y in the script. Template update
        step will not use the rigid component if this is True.

    output_dir : path
        directory name where intermediate transforms are written

    kwargs : keyword args
        extra arguments passed to ants registration

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image2 = ants.image_read( ants.get_ants_data('r27') )
    >>> image3 = ants.image_read( ants.get_ants_data('r85') )
    >>> timage = ants.build_template( image_list = ( image, image2, image3 ) ).resample_image( (45,45))
    >>> timagew = ants.build_template( image_list = ( image, image2, image3 ), weights = (5,1,1) )
    """
    work_dir = mktemp() if output_dir is None else output_dir

    def make_outprefix(k: int):
        os.makedirs(os.path.join(work_dir, f"img{k:04d}"), exist_ok=True)
        return os.path.join(work_dir, f"img{k:04d}", "out")

    if "type_of_transform" not in kwargs:
        type_of_transform = "SyN"
    else:
        type_of_transform = kwargs.pop("type_of_transform")

    if weights is None:
        weights = np.repeat(1.0 / len(image_list), len(image_list))
    weights = [x / sum(weights) for x in weights]
    if initial_template is None:
        initial_template = image_list[0] * 0
        for i in range(len(image_list)):
            temp = image_list[i] * weights[i]
            temp = ants.resample_image_to_target(temp, initial_template)
            initial_template = initial_template + temp

    xavg = initial_template.clone()
    for i in range(iterations):
        affinelist = []
        for k in range(len(image_list)):
            w1 = ants.registration(
                xavg, image_list[k], type_of_transform=type_of_transform, outprefix=make_outprefix(k), **kwargs
            )
            L = len(w1["fwdtransforms"])
            # affine is the last one
            affinelist.append(w1["fwdtransforms"][L-1])

            if k == 0:
                if L == 2:
                    wavg = ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = w1["warpedmovout"] * weights[k]
            else:
                if L == 2:
                    wavg = wavg + ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[k]

        if useNoRigid:
            avgaffine = ants.average_affine_transform_no_rigid(affinelist)
        else:
            avgaffine = ants.average_affine_transform(affinelist)
        afffn = os.path.join(work_dir, "avgAffine.mat")
        ants.write_transform(avgaffine, afffn)

        if L == 2:
            print(wavg.abs().mean())
            wscl = (-1.0) * gradient_step
            wavg = wavg * wscl
            # apply affine to the nonlinear?
            # need to save the average
            wavgA = ants.apply_transforms(fixed=xavgNew, moving=wavg, imagetype=1, transformlist=afffn, whichtoinvert=[1])
            wavgfn = os.path.join(work_dir, "avgWarp.nii.gz")
            ants.image_write(wavgA, wavgfn)
            xavg = ants.apply_transforms(fixed=xavgNew, moving=xavgNew, transformlist=[wavgfn, afffn], whichtoinvert=[0, 1])
        else:
            xavg = ants.apply_transforms(fixed=xavgNew, moving=xavgNew, transformlist=[afffn], whichtoinvert=[1])
            
        if blending_weight is not None:
            xavg = xavg * blending_weight + ants.iMath(xavg, "Sharpen") * (
                1.0 - blending_weight
            )

    if output_dir is None:
        shutil.rmtree(work_dir)
    return xavg
