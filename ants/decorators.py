from functools import wraps
from ants.core.ants_image import ANTsImage

def image_method(func):
    @wraps(func) 
    def wrapper(self, *args, **kwargs): 
        return func(self, *args, **kwargs)
    setattr(ANTsImage, func.__name__, wrapper)
    return func
