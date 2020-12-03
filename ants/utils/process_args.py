"""
"""


__all__ = [
    "get_pointer_string",
    "short_ptype",
    "_ptrstr",
    "_int_antsProcessArguments",
    "get_lib_fn",
]

from ..core import ants_image as iio
from .. import lib

_short_ptype_map = {
    "unsigned char": "UC",
    "unsigned int": "UI",
    "float": "F",
    "double": "D",
}


def short_ptype(pixeltype):
    return _short_ptype_map[pixeltype]


def get_lib_fn(string):
    return lib.__dict__[string]


def _ptrstr(pointer):
    """ get string representation of a py::capsule (aka pointer) """
    libfn = get_lib_fn("ptrstr")
    return libfn(pointer)


def get_pointer_string(image):
    return _ptrstr(image.pointer)


def _int_antsProcessArguments(args):
    """
    Needs to be better validated.
    """
    p_args = []
    if isinstance(args, dict):
        for argname, argval in args.items():
            if "-MULTINAME-" in argname:
                # have this little hack because python doesnt support
                # multiple dict entries w/ the same key like R lists
                argname = argname[: argname.find("-MULTINAME-")]
            if argval is not None:
                if len(argname) > 1:
                    p_args.append("--%s" % argname)
                else:
                    p_args.append("-%s" % argname)

                if isinstance(argval, iio.ANTsImage):
                    p_args.append(_ptrstr(argval.pointer))
                elif isinstance(argval, list):
                    for av in argval:
                        if isinstance(av, iio.ANTsImage):
                            av = _ptrstr(av.pointer)
                        elif str(arg) == "True":
                            av = str(1)
                        elif str(arg) == "False":
                            av = str(0)
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
            elif str(arg) == "True":
                p_arg = str(1)
            elif str(arg) == "False":
                p_arg = str(0)
            else:
                p_arg = str(arg)
            p_args.append(p_arg)
    return p_args
