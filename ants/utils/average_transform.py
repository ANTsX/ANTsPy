from .. import utils, core
from tempfile import mktemp
import os



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
    tf = core.ants_transform_io.read_transform(transformlist[0])
    if referencetransform is None:
        args = [tf.dimension, res_temp_file] + transformlist
    else:
        args = [tf.dimension, res_temp_file] + ['-R',  referencetransform] + transformlist
    pargs = utils._int_antsProcessArguments(args)
    print(pargs)
    libfun = utils.get_lib_fn(funcname)
    status = libfun(pargs)

    res = core.ants_transform_io.read_transform(res_temp_file)
    os.remove(res_temp_file)
    return res

def average_affine_transform(transformlist, referencetransform=None):
    return _average_affine_transform_driver(transformlist, referencetransform, "AverageAffineTransform")


def average_affine_transform_no_rigid(transformlist, referencetransform=None):
    return _average_affine_transform_driver(transformlist, referencetransform, "AverageAffineTransformNoRigid")
    
