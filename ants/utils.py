from . import lib
from .core import ants_image as iio

_short_ptype_map = {
    "unsigned char": "UC",
    "unsigned int": "UI",
    "float": "F",
    "double": "D",
}

def short_ptype(pixeltype):
    return _short_ptype_map[pixeltype]


def ptrstr(pointer):
    """ get string representation of a py::capsule (aka pointer) """
    return lib.ptrstr(pointer)


def process_arguments(args):
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
                    if argname == 'verbose':
                        p_args.append('-v')
                        argval = 1
                    else:
                        p_args.append("--%s" % argname)
                else:
                    p_args.append("-%s" % argname)

                if isinstance(argval, iio.ANTsImage):
                    p_args.append(ptrstr(argval.pointer))
                elif isinstance(argval, list):
                    for av in argval:
                        if isinstance(av, iio.ANTsImage):
                            av = ptrstr(av.pointer)
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
                pointer_string = ptrstr(arg.pointer)
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
