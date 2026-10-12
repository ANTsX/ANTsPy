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

def _short_ptype(pixeltype):
    return _short_ptype_map[pixeltype]


def _short_dtype(dtype):
    return _short_dtype_map[dtype]


def _get_best_supported_dtype(input_dtype):
    """
    Get the supported NumPy dtype for a given input dtype. This is not a "safe" cast and takes into account
    the internal requirements of ANTsPy.

    Input NumPy dtype    Output NumPy dtype  Precision
    ===================  ==================  ==================
    bool                 uint8               Exact
    uint8                uint8               Exact
    int8                 float32             Exact
    int16                float32             Exact
    uint16               uint32              Exact
    int32                float32             May lose precision
    uint32               uint32              Exact
    int64                float32             May lose precision
    float16              float32             Exact
    float32              float32             Exact
    float64              float32             May lose range/precision
    Other (incl. uint64) None                --
    ===================  ==================  ===================

    Args:
        input_dtype (numpy.dtype or str): The input NumPy dtype.

    Returns:
        str: The best supported NumPy dtype as a string, or None if no supported dtype is found.
    """
    best_dtype = str(input_dtype)

    # Convert float64 to float32
    # This was originally intended to be temporary, but many functions now expect it
    # numpy uses float64 extensively, many instances of unwanted promotion to float64 would
    # occur if this was removed
    if best_dtype == 'float64':
        best_dtype = 'float32'

    # if dtype is not supported, cast to best available if possible
    if best_dtype not in _supported_dtypes:
        best_dtype = _unsupported_dtype_map.get(best_dtype)

    return best_dtype
