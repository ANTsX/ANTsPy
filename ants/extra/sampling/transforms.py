"""
Various data augmentation transforms for ANTsImage types

List of Transformations:
======================
- CastIntensity
- BlurIntensity
- NormalizeIntensity
- RescaleIntensity
- ShiftScaleIntensity
- SigmoidIntensity
======================
- FlipImage
- TranslateImage

TODO
----
- RotateImage
- ShearImage
- ScaleImage
- DeformImage
- PadImage
- HistogramEqualizeIntensity
- TruncateIntensity
- SharpenIntensity
- MorpholigicalIntensity
    - MD
    - ME
    - MO
    - MC
    - GD
    - GE
    - GO
    - GC
"""
__all__ = ['CastIntensity',
           'BlurIntensity',
           'LocallyBlurIntensity',
           'NormalizeIntensity',
           'RescaleIntensity',
           'ShiftScaleIntensity',
           'SigmoidIntensity',
           'FlipImage',
           'ScaleImage',
           'TranslateImage',
           'MultiResolutionImage']

from ... import utils
from ...core import ants_image as iio


class MultiResolutionImage(object):
    """
    Generate a set of images at multiple resolutions from an original image
    """
    def __init__(self, levels=4, keep_shape=False):
        self.levels = levels
        self.keep_shape = keep_shape

    def transform(self, X, y=None):
        """
        Generate a set of multi-resolution ANTsImage types

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform

        Example
        -------
        >>> import ants
        >>> multires = ants.contrib.MultiResolutionImage(levels=4)
        >>> img = ants.image_read(ants.get_data('r16'))
        >>> imgs = multires.transform(img)
        """
        insuffix = X._libsuffix
        multires_fn = utils.get_lib_fn('multiResolutionAntsImage%s' % (insuffix))
        casted_ptrs = multires_fn(X.pointer, self.levels)
        
        imgs = []
        for casted_ptr in casted_ptrs:
            img = iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                                components=X.components, pointer=casted_ptr)
            if self.keep_shape:
                img = img.resample_image_to_target(X)
            imgs.append(img)

        return imgs


## Intensity Transforms ##

class CastIntensity(object):
    """
    Cast the pixeltype of an ANTsImage to a given type. 
    This code uses the C++ ITK library directly, so it is fast.
    
    NOTE: This offers a ~2.5x speedup over using img.clone(pixeltype):
    
    Timings vs Cloning
    ------------------
    >>> import ants
    >>> import time
    >>> caster = ants.contrib.CastIntensity('float')
    >>> img = ants.image_read(ants.get_data('mni')).clone('unsigned int')
    >>> s = time.time()
    >>> for i in range(1000):
    ...     img_float = caster.transform(img)
    >>> e = time.time()
    >>> print(e - s) # 9.6s
    >>> s = time.time()
    >>> for i in range(1000):
    ...     img_float = img.clone('float')
    >>> e = time.time()
    >>> print(e - s) # 25.3s
    """
    def __init__(self, pixeltype):
        """
        Initialize a CastIntensity transform
    
        Arguments
        ---------
        pixeltype : string
            pixeltype to which images will be casted

        Example
        -------
        >>> import ants
        >>> caster = ants.contrib.CastIntensity('float')
        """
        self.pixeltype = pixeltype

    def transform(self, X, y=None):
        """
        Transform an image by casting its type

        Arguments
        ---------
        X : ANTsImage
            image to cast

        y : ANTsImage (optional)
            another image to cast.

        Example
        -------
        >>> import ants
        >>> caster = ants.contrib.CastIntensity('float')
        >>> img2d = ants.image_read(ants.get_data('r16')).clone('unsigned int')
        >>> img2d_float = caster.transform(img2d)
        >>> print(img2d.pixeltype, '- ', img2d_float.pixeltype)
        >>> img3d = ants.image_read(ants.get_data('mni')).clone('unsigned int')
        >>> img3d_float = caster.transform(img3d)
        >>> print(img3d.pixeltype, ' - ' , img3d_float.pixeltype)
        """
        insuffix = X._libsuffix
        outsuffix = '%s%i' % (utils.short_ptype(self.pixeltype), X.dimension)
        cast_fn = utils.get_lib_fn('castAntsImage%s%s' % (insuffix, outsuffix))
        casted_ptr = cast_fn(X.pointer)
        return iio.ANTsImage(pixeltype=self.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)


class BlurIntensity(object):
    """
    Transform for blurring the intensity of an ANTsImage
    using a Gaussian Filter
    """
    def __init__(self, sigma, width):
        """
        Initialize a BlurIntensity transform

        Arguments
        ---------
        sigma : float
            variance of gaussian kernel intensity
            increasing this value increasing the amount
            of blur

        width : int
            width of gaussian kernel shape
            increasing this value increase the number of
            neighboring voxels which are used for blurring

        Example
        -------
        >>> import ants
        >>> blur = ants.contrib.BlurIntensity(2,3)
        """
        self.sigma = sigma
        self.width = width

    def transform(self, X, y=None):
        """
        Blur an image by applying a gaussian filter.

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform.

        Example
        -------
        >>> import ants
        >>> blur = ants.contrib.BlurIntensity(2,3)
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_b = blur.transform(img2d)
        >>> ants.plot(img2d)
        >>> ants.plot(img2d_b)
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_b = blur.transform(img3d)
        >>> ants.plot(img3d)
        >>> ants.plot(img3d_b)
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('blurAntsImage%s' % (insuffix))
        casted_ptr = cast_fn(X.pointer, self.sigma, self.width)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr,
                             origin=X.origin)


class LocallyBlurIntensity(object):
    """
    Blur an ANTsImage locally using a gradient anisotropic
    diffusion filter, thereby preserving the sharpeness of edges as best 
    as possible.
    """
    def __init__(self, conductance=1, iters=5):
        self.conductance = conductance
        self.iters = iters

    def transform(self, X, y=None):
        """
        Locally blur an image by applying a gradient anisotropic diffusion filter.

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform.

        Example
        -------
        >>> import ants
        >>> blur = ants.contrib.LocallyBlurIntensity(1,5)
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_b = blur.transform(img2d)
        >>> ants.plot(img2d)
        >>> ants.plot(img2d_b)
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_b = blur.transform(img3d)
        >>> ants.plot(img3d)
        >>> ants.plot(img3d_b)
        """
        #if X.pixeltype != 'float':
        #    raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')
        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('locallyBlurAntsImage%s' % (insuffix))
        casted_ptr = cast_fn(X.pointer, self.iters, self.conductance)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)


class NormalizeIntensity(object):
    """
    Normalize the intensity values of an ANTsImage to have
    zero mean and unit variance

    NOTE: this transform is more-or-less the same in speed
    as an equivalent numpy+scikit-learn solution.

    Timing vs Numpy+Scikit-Learn
    ----------------------------
    >>> import ants
    >>> import numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    >>> import time
    >>> img = ants.image_read(ants.get_data('mni'))
    >>> arr = img.numpy().reshape(1,-1)
    >>> normalizer = ants.contrib.NormalizeIntensity()
    >>> normalizer2 = StandardScaler()
    >>> s = time.time()
    >>> for i in range(100):
    ...     img_scaled = normalizer.transform(img)
    >>> e = time.time()
    >>> print(e - s) # 3.3s
    >>> s = time.time()
    >>> for i in range(100):
    ...     arr_scaled = normalizer2.fit_transform(arr)
    >>> e = time.time()
    >>> print(e - s) # 3.5s
    """
    def __init__(self):
        """
        Initialize a NormalizeIntensity transform
        """
        pass

    def transform(self, X, y=None):
        """
        Transform an image by normalizing its intensity values to 
        have zero mean and unit variance.

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform.

        Example
        -------
        >>> import ants
        >>> normalizer = ants.contrib.NormalizeIntensity()
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = normalizer.transform(img2d)
        >>> print(img2d.mean(), ',', img2d.std(), ' -> ', img2d_r.mean(), ',', img2d_r.std())
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_r = normalizer.transform(img3d)
        >>> print(img3d.mean(), ',' , img3d.std(), ',', ' -> ', img3d_r.mean(), ',' , img3d_r.std())  
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('normalizeAntsImage%s' % (insuffix))
        casted_ptr = cast_fn(X.pointer)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)


class RescaleIntensity(object):
    """
    Rescale the pixeltype of an ANTsImage linearly to be between a given
    minimum and maximum value. 
    This code uses the C++ ITK library directly, so it is fast.
    
    NOTE: this offered a ~5x speedup over using built-in arithmetic operations in ANTs.
    It is also more-or-less the same in speed as an equivalent numpy+scikit-learn
    solution.

    Timing vs Built-in Operations
    -----------------------------
    >>> import ants
    >>> import time
    >>> rescaler = ants.contrib.RescaleIntensity(0,1)
    >>> img = ants.image_read(ants.get_data('mni'))
    >>> s = time.time()
    >>> for i in range(100):
    ...     img_float = rescaler.transform(img)
    >>> e = time.time()
    >>> print(e - s) # 2.8s
    >>> s = time.time()
    >>> for i in range(100):
    ...     maxval = img.max()
    ...     img_float = (img - maxval) / (maxval - img.min())
    >>> e = time.time()
    >>> print(e - s) # 13.9s

    Timing vs Numpy+Scikit-Learn
    ----------------------------
    >>> import ants
    >>> import numpy as np
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> import time
    >>> img = ants.image_read(ants.get_data('mni'))
    >>> arr = img.numpy().reshape(1,-1)
    >>> rescaler = ants.contrib.RescaleIntensity(-1,1)
    >>> rescaler2 = MinMaxScaler((-1,1)).fit(arr)
    >>> s = time.time()
    >>> for i in range(100):
    ...     img_scaled = rescaler.transform(img)
    >>> e = time.time()
    >>> print(e - s) # 2.8s
    >>> s = time.time()
    >>> for i in range(100):
    ...     arr_scaled = rescaler2.transform(arr)
    >>> e = time.time()
    >>> print(e - s) # 3s
    """

    def __init__(self, min_val, max_val):
        """
        Initialize a RescaleIntensity transform.

        Arguments
        ---------
        min_val : float
            minimum value to which image(s) will be rescaled

        max_val : float
            maximum value to which image(s) will be rescaled
        
        Example
        -------
        >>> import ants
        >>> rescaler = ants.contrib.RescaleIntensity(0,1)
        """
        self.min_val = min_val
        self.max_val = max_val

    def transform(self, X, y=None):
        """
        Transform an image by linearly rescaling its intensity to
        be between a minimum and maximum value

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform.

        Example
        -------
        >>> import ants
        >>> rescaler = ants.contrib.RescaleIntensity(0,1)
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = rescaler.transform(img2d)
        >>> print(img2d.min(), ',', img2d.max(), ' -> ', img2d_r.min(), ',', img2d_r.max())
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_r = rescaler.transform(img3d)
        >>> print(img3d.min(), ',' , img3d.max(), ' -> ', img3d_r.min(), ',' , img3d_r.max())
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('rescaleAntsImage%s' % (insuffix))
        casted_ptr = cast_fn(X.pointer, self.min_val, self.max_val)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)


class ShiftScaleIntensity(object):
    """
    Shift and scale the intensity of an ANTsImage
    """
    def __init__(self, shift, scale):
        """
        Initialize a ShiftScaleIntensity transform

        Arguments
        ---------
        shift : float
            shift all of the intensity values by the given amount through addition.
            For example, if the minimum image value is 0.0 and the shift
            is 10.0, then the new minimum value (before scaling) will be 10.0

        scale : float
            scale all the intensity values by the given amount through multiplication.
            For example, if the min/max image values are 10/20 and the scale
            is 2.0, then then new min/max values will be 20/40

        Example
        -------
        >>> import ants
        >>> shiftscaler = ants.contrib.ShiftScaleIntensity(shift=10, scale=2)
        """
        self.shift = shift
        self.scale = scale

    def transform(self, X, y=None):
        """
        Transform an image by shifting and scaling its intensity values.

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform.

        Example
        -------
        >>> import ants
        >>> shiftscaler = ants.contrib.ShiftScaleIntensity(10,2.)
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = shiftscaler.transform(img2d)
        >>> print(img2d.min(), ',', img2d.max(), ' -> ', img2d_r.min(), ',', img2d_r.max())
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_r = shiftscaler.transform(img3d)
        >>> print(img3d.min(), ',' , img3d.max(), ',', ' -> ', img3d_r.min(), ',' , img3d_r.max())  
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('shiftScaleAntsImage%s' % (insuffix))
        casted_ptr = cast_fn(X.pointer, self.scale, self.shift)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)


class SigmoidIntensity(object):
    """
    Transform an image using a sigmoid function
    """
    def __init__(self, min_val, max_val, alpha, beta):
        """
        Initialize a SigmoidIntensity transform

        Arguments
        ---------
        min_val : float
            minimum value

        max_val : float
            maximum value

        alpha : float
            alpha value for sigmoid

        beta : flaot
            beta value for sigmoid

        Example
        -------
        >>> import ants
        >>> sigscaler = ants.contrib.SigmoidIntensity(0,1,1,1)
        """
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha
        self.beta = beta

    def transform(self, X, y=None):
        """
        Transform an image by applying a sigmoid function.

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform.

        Example
        -------
        >>> import ants
        >>> sigscaler = ants.contrib.SigmoidIntensity(0,1,1,1)
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = sigscaler.transform(img2d)
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_r = sigscaler.transform(img3d)
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('sigmoidAntsImage%s' % (insuffix))
        casted_ptr = cast_fn(X.pointer, self.min_val, self.max_val, self.alpha, self.beta)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)


## Physical Transforms ##

class FlipImage(object):
    """
    Transform an image by flipping two axes. 
    """
    def __init__(self, axis1, axis2):
        """
        Initialize a SigmoidIntensity transform

        Arguments
        ---------
        axis1 : int
            axis to flip

        axis2 : int
            other axis to flip

        Example
        -------
        >>> import ants
        >>> flipper = ants.contrib.FlipImage(0,1)
        """
        self.axis1 = axis1
        self.axis2 = axis2

    def transform(self, X, y=None):
        """
        Transform an image by applying a sigmoid function.

        Arguments
        ---------
        X : ANTsImage
            image to transform

        y : ANTsImage (optional)
            another image to transform.

        Example
        -------
        >>> import ants
        >>> flipper = ants.contrib.FlipImage(0,1)
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = flipper.transform(img2d)
        >>> ants.plot(img2d)
        >>> ants.plot(img2d_r)
        >>> flipper2 = ants.contrib.FlipImage(1,0)
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = flipper2.transform(img2d)
        >>> ants.plot(img2d)
        >>> ants.plot(img2d_r)
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('flipAntsImage%s' % (insuffix))
        casted_ptr = cast_fn(X.pointer, self.axis1, self.axis2)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr,
                             origin=X.origin)


class TranslateImage(object):
    """
    Translate an image in physical space. This function calls 
    highly optimized ITK/C++ code.
    """
    def __init__(self, translation, reference=None, interp='linear'):
        """
        Initialize a TranslateImage transform

        Arguments
        ---------
        translation : list, tuple, or numpy.ndarray
            absolute pixel transformation in each axis

        reference : ANTsImage (optional)
            image which provides the reference physical space in which
            to perform the transform

        interp : string
            type of interpolation to use
            options: linear, nearest

        Example
        -------
        >>> import ants
        >>> translater = ants.contrib.TranslateImage((10,10), interp='linear')
        """
        if interp not in {'linear', 'nearest'}:
            raise ValueError('interp must be one of {linear, nearest}')

        self.translation = list(translation)
        self.reference = reference
        self.interp = interp

    def transform(self, X, y=None):
        """
        Example
        -------
        >>> import ants
        >>> translater = ants.contrib.TranslateImage((40,0))
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = translater.transform(img2d)
        >>> ants.plot(img2d, img2d_r)
        >>> translater = ants.contrib.TranslateImage((40,0,0))
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_r = translater.transform(img3d)
        >>> ants.plot(img3d, img3d_r, axis=2)
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        if len(self.translation) != X.dimension:
            raise ValueError('must give a translation value for each image dimension')

        if self.reference is None:
            reference = X
        else:
            reference = self.reference

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('translateAntsImage%s_%s' % (insuffix, self.interp))
        casted_ptr = cast_fn(X.pointer, reference.pointer, self.translation)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)


class ScaleImage(object):
    """
    Scale an image in physical space. This function calls 
    highly optimized ITK/C++ code.
    """
    def __init__(self, scale, reference=None, interp='linear'):
        """
        Initialize a TranslateImage transform

        Arguments
        ---------
        scale : list, tuple, or numpy.ndarray
            relative scaling along each axis

        reference : ANTsImage (optional)
            image which provides the reference physical space in which
            to perform the transform

        interp : string
            type of interpolation to use
            options: linear, nearest

        Example
        -------
        >>> import ants
        >>> translater = ants.contrib.TranslateImage((10,10), interp='linear')
        """
        if interp not in {'linear', 'nearest'}:
            raise ValueError('interp must be one of {linear, nearest}')

        self.scale = list(scale)
        self.reference = reference
        self.interp = interp

    def transform(self, X, y=None):
        """
        Example
        -------
        >>> import ants
        >>> scaler = ants.contrib.ScaleImage((1.2,1.2))
        >>> img2d = ants.image_read(ants.get_data('r16'))
        >>> img2d_r = scaler.transform(img2d)
        >>> ants.plot(img2d, img2d_r)
        >>> scaler = ants.contrib.ScaleImage((1.2,1.2,1.2))
        >>> img3d = ants.image_read(ants.get_data('mni'))
        >>> img3d_r = scaler.transform(img3d)
        >>> ants.plot(img3d, img3d_r)
        """
        if X.pixeltype != 'float':
            raise ValueError('image.pixeltype must be float ... use TypeCast transform or clone to float')

        if len(self.scale) != X.dimension:
            raise ValueError('must give a scale value for each image dimension')

        if self.reference is None:
            reference = X
        else:
            reference = self.reference

        insuffix = X._libsuffix
        cast_fn = utils.get_lib_fn('scaleAntsImage%s_%s' % (insuffix, self.interp))
        casted_ptr = cast_fn(X.pointer, reference.pointer, self.scale)
        return iio.ANTsImage(pixeltype=X.pixeltype, dimension=X.dimension,
                             components=X.components, pointer=casted_ptr)

