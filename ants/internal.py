import ants
from ants import lib

short_ptype_map = {"unsigned char": "UC",
                    "unsigned int": "UI",
                    "float": "F",
                    "double": "D"}

def infer_dtype(dtype):
    # supported dtypes: uint8, uint32, float32, float64
    exchange_map = {
        'bool': 'uint8',
        'int8': 'uint32',
        'int16': 'uint32',
        'int32': 'uint32',
        'int64': 'uint32',
        'uint16': 'uint32',
        'uint64': 'uint32',
        'float16': 'float32'
    }
    new_dtype = exchange_map.get(str(dtype), dtype)
    return new_dtype

def short_ptype(pixeltype):
    return short_ptype_map[pixeltype]

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
