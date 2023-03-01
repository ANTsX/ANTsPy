"""
"""


__all__ = [
    "ANTsSerializer",
    "short_ptype",
    "get_lib_fn",
]

import platform
import os
import atexit
from ..core import ants_image as iio
from .. import lib

_short_ptype_map = {
    "unsigned char": "UC",
    "unsigned int": "UI",
    "float": "F",
    "double": "D",
}

temporary_folder = None

def temp_dir():
    global temporary_folder
    if temporary_folder is None or not os.path.exists(temporary_folder):
        temporary_folder = tempfile.mkdtemp(prefix="ANTsPyTmp")
    return temporary_folder

def temp_file(suffix=None, prefix=None, dir=None):
    if dir is None:
        dir = temp_dir()
    return tempfile.mktemp(suffix=suffix, prefix=prefix, dir=dir)

@atexit.register
def clear_temp_dir():
    if temporary_folder is not None and os.path.exists(temporary_folder):
        os.removedirs(temporary_folder)


def short_ptype(pixeltype):
    return _short_ptype_map[pixeltype]


def get_lib_fn(string):
    return lib.__dict__[string]

def _ptrstr(pointer):
    """ get string representation of a py::capsule (aka pointer) """
    libfn = get_lib_fn("ptrstr")
    return libfn(pointer)

class ANTsSerializer(object):
    def __init__(self):
        self._temp_files = []
        self._os = platform.system()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        # There is no need to handle exception type
        for tmpfile in self._temp_files:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
        # clear the list, might be overkilling
        self._temp_files.clear();
    
    def get_pointer_string(self, image):
        if self._os == "Windows":
            # save to a temporary file
            tfile = temp_file(suffix=".nii.gz", prefix="ANTsSerializer")
            # just to be safe in case backslashes are not properly quoted
            tfile = os.path.normpath(tfile).replace(os.sep, '/')
            # add this file first in case the file is created before errors
            self._temp_files.append(tfile)
            image.to_file(tfile)
        else:
            tfile = _ptrstr(image.pointer)
        return tfile
    
    def int_antsProcessArguments(self, args):
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
                        p_args.append(self.get_pointer_string(argval))
                    elif isinstance(argval, list):
                        for av in argval:
                            if isinstance(av, iio.ANTsImage):
                                av = self.get_pointer_string(av)
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
                    pointer_string = self.get_pointer_string(arg)
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
    

