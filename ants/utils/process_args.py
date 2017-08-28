"""
"""

 

__all__ = ['get_pointer_string',
           '_ptrstr', 
           '_int_antsProcessArguments']

from ..core import ants_image as iio
from .. import lib


def _ptrstr(pointer):
    """ get string representation of a py::capsule (aka pointer) """
    return lib.ptrstr(pointer)

def get_pointer_string(img):
    return _ptrstr(img.pointer)

def _int_antsProcessArguments(args):
    """
    Needs to be better validated.
    """
    p_args = []
    if isinstance(args, dict):
        for argname, argval in args.items():
            if '-MULTINAME-' in argname:
                # have this little hack because python doesnt support
                # multiple dict entries w/ the same key like R lists
                argname = argname[:argname.find('-MULTINAME-')]
            if argval is not None:
                if len(argname) > 1:
                    p_args.append('--%s' % argname)
                else:
                    p_args.append('-%s' % argname)

                if isinstance(argval, iio.ANTsImage):
                    p_args.append(_ptrstr(argval.pointer))
                elif isinstance(argval, list):
                    for av in argval:
                        p_args.append(av)
                else:
                    p_args.append(str(argval))

    elif isinstance(args, list):
        for arg in args:
            if isinstance(arg, iio.ANTsImage):
                pointer_string = _ptrstr(arg.pointer)
                p_arg = pointer_string
            elif arg is None:
                pass
            else:
                p_arg = str(arg)
            p_args.append(p_arg)
    return p_args


