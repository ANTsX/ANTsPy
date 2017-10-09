"""
Various data augmentation transforms for ANTsImage types

List of Transformations:
- TypeCast

"""

from ... import utils
from ...core import ants_image as iio



class TypeCast(object):
    """
    Cast the pixeltype of an ANTsImage to a given type. 
    This code uses the C++ ITK library directly, so is fast.
    """
    def __init__(self, pixeltype):
        """
        Initialize a TypeCast transform
        """
        self.pixeltype = pixeltype

    def transform(self, X, y):
        insuffix = X._libsuffix
        outsuffix = '%s%i' % (utils.short_ptype(self.pixeltype), X.dimension)
        cast_fn = utils.get_lib_fn('castAntsImage%s%s' % (insuffix, outsuffix))
        casted_ptr = cast_fn(X.pointer)
        return iio.ANTsImage(pixeltype=self.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)