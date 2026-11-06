import ants
from ants import lib
import numpy as np

def get_lib_fn(string):
    return getattr(lib, string)

def get_pointer_string(image):
    return lib.ptrstr(image.pointer)

def process_arguments(args):
    p_args = []
    if isinstance(args, dict):
        for argname, argval in args.items():

            # -MULTINAME- is used to support multiple items with the same key
            if "-MULTINAME-" in argname:
                argname = argname[: argname.find("-MULTINAME-")]

            if argval is not None:
                if len(argname) > 1:
                    p_args.append("--%s" % argname)
                else:
                    p_args.append("-%s" % argname)

                if ants.is_image(argval):
                    p_args.append(get_pointer_string(argval))
                elif isinstance(argval, list):
                    for av in argval:
                        if ants.is_image(av):
                            av = get_pointer_string(av)
                        elif str(arg) == "True":
                            av = str(1)
                        elif str(arg) == "False":
                            av = str(0)
                        p_args.append(av)
                else:
                    p_args.append(str(argval))

    elif isinstance(args, list):
        for arg in args:
            if ants.is_image(arg):
                pointer_string = get_pointer_string(arg)
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


def is_real_scalar(value):
    '''
    Check if a value is a real scalar (int or float, but not bool). Works for both Python built-in types and NumPy scalar types.
    '''
    return (
        isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    )


def is_integer(value):
    '''
    Check if a value is an integer (int or np.integer, but not bool). Works for both Python built-in types and NumPy scalar types.
    '''
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))