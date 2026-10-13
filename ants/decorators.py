import ants
from functools import wraps
from ants.core.ants_image import ANTsImage

def image_method(func):
    @wraps(func) 
    def wrapper(self, *args, **kwargs): 
        return func(self, *args, **kwargs)
    setattr(ANTsImage, func.__name__, wrapper)
    return func


def components_method(func):
    @wraps(func) 
    def wrapper(image, *args, **kwargs): 
        if image.has_components:
            return ants.merge_channels([func(img, *args, **kwargs) for img in ants.split_channels(image)],
                                       channels_first=image.channels_first)
        else:
            return func(image, *args, **kwargs)
    return wrapper
