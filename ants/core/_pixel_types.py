"""

Shared pixel-type definitions

Following ants_image, we use 'dtype' for a numpy dtype (eg, 'uint8') and 'ptype' for the ITK pixel type (eg, 'unsigned char').

"""

_supported_ptypes = ("unsigned char", "unsigned int", "float", "double")
_supported_dtypes = ("uint8", "uint32", "float32", "float64")

# Unsupported_ptypes include "char", "unsigned short", "short", "int".
# These will be cast if possible, ie if a mapping exists in _unsupported_ptype_map

_unsupported_ptype_map = {
    "char": "float",
    "unsigned short": "unsigned int",
    "short": "float",
    "int": "float",
}

_unsupported_dtype_map = {
        "bool": "uint8",
        "int8": "float32",
        "int16": "float32",
        "uint16": "uint32",
        "float16": "float32",
        "int32": "float32",
        "int64": "float32",
}

# _ptype_type_map = {
# Used to map pixel types to their short forms for C++ functions
_short_ptype_map = {
    "unsigned char": "UC",
    "unsigned int": "UI",
    "float": "F",
    "double": "D",
}

# _dtype_type_map
_short_dtype_map = {
    "uint8": "UC",
    "uint32": "UI",
    "float32": "F",
    "float64": "D"
}

# Map of supported numpy dtypes to ITK pixel types
_npy_to_itk_map = {
    "uint8": "unsigned char",
    "uint32": "unsigned int",
    "float32": "float",
    "float64": "double",
}
_itk_to_npy_map = {v: k for k, v in _npy_to_itk_map.items()}

def short_ptype(pixeltype):
    return _short_ptype_map[pixeltype]

def short_dtype(dtype):
    return _short_dtype_map[dtype]