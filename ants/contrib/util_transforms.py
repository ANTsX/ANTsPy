"""
Utility and core transforms
"""


class Compose(object):
    """
    Compose a set of transforms together sequentially or 
    in parallel applied to one or more images.
    
    Note that this class does NOT combine transforms together,
    but rather applies them individually.
    """
    def __init__(self, transforms):
        pass

    def transform(self, X, y=None):
        pass