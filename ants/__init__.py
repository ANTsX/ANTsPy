
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("antspyx")
except _PackageNotFoundError:
    # Documentation builds import the source tree without installing antspyx.
    __version__ = "unknown"

from .core import *
from .label import *
from .learn import *
from .math import *
from .ops import *
from .plotting import *
from .registration import *
from .segmentation import *
from .utils import *
from .contrib import *
from .deeplearn import *
