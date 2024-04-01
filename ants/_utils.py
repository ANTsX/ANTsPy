from . import lib
from .core.ants_image import AntsImage


def short_ptype(pixeltype):
    _short_ptype_map = {
        "unsigned char": "UC",
        "unsigned int": "UI",
        "float": "F",
        "double": "D",
    }
    return _short_ptype_map[pixeltype]

def ptrstr(pointer):
    """ get string representation of pointer to underlying ants image """
    return lib.ptrstr(pointer)


def process_arguments(args):
    """Turn a dictionary/list of python arguments into ANTs-style arguments"""
    p_args = []
    if isinstance(args, dict):
        for argname, argval in args.items():
            if "-MULTINAME-" in argname:
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

                if isinstance(argval, AntsImage):
                    p_args.append(ptrstr(argval.pointer))
                elif isinstance(argval, list):
                    for av in argval:
                        if isinstance(av, AntsImage):
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
            if isinstance(arg, AntsImage):
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
