
from tempfile import mktemp
import os

import ants
from ants.internal import get_lib_fn, process_arguments

__all__ = ['average_affine_transform', 
           'average_affine_transform_no_rigid']


def _average_affine_transform_driver(transformlist, referencetransform=None, funcname="AverageAffineTransform"):
    """
    takes a list of transforms (files at the moment)
    and returns the average
    """

    # AverageAffineTransform deals with transform files,
    # so this function will need to deal with already
    # loaded files. Doesn't look like the magic
    # available for images has been added for transforms.
    res_temp_file = mktemp(suffix='.mat')

    # could do some stuff here to cope with transform lists that
    # aren't files
    
    # load one of the transforms to figure out the dimension
    tf = ants.read_transform(transformlist[0])
    if referencetransform is None:
        args = [tf.dimension, res_temp_file] + transformlist
    else:
        args = [tf.dimension, res_temp_file] + ['-R',  referencetransform] + transformlist
    pargs = process_arguments(args)
    print(pargs)
    libfun = get_lib_fn(funcname)
    status = libfun(pargs)

    res = ants.read_transform(res_temp_file)
    os.remove(res_temp_file)
    return res

def average_affine_transform(transformlist, referencetransform=None):
    return _average_affine_transform_driver(transformlist, referencetransform, "AverageAffineTransform")


def average_affine_transform_no_rigid(transformlist, referencetransform=None):
    return _average_affine_transform_driver(transformlist, referencetransform, "AverageAffineTransformNoRigid")
    
