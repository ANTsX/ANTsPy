"""
ANTs ImageToImageMetric class
"""

__all__ = []


from . import ants_image as iio


class ANTsImageToImageMetric(object):
    """
    ANTsImageToImageMetric class
    """

    def __init__(self, metric):
        self._metric = metric
        self._is_initialized = False

    # ------------------------------------------
    # PROPERTIES
    @property
    def precision(self):
        return self._metric.precision

    @property
    def dimension(self):
        return self._metric.dimension

    @property
    def metrictype(self):
        return self._metric.metrictype.replace('ImageToImageMetricv4','')

    @property
    def is_vector(self):
        return self._metric.isVector == 1

    @property
    def pointer(self):
        return self._metric.pointer

    # ------------------------------------------
    # METHODS
    def set_fixed_image(self, image):
        """
        Set Fixed ANTsImage for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setFixedImage(image.pointer, False)

    def set_fixed_mask(self, image):
        """
        Set Fixed ANTsImage Mask for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setFixedImage(image.pointer, True)

    def set_moving_image(self, image):
        """
        Set Moving ANTsImage for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setMovingImage(image.pointer, False)

    def set_moving_mask(self, image):
        """
        Set Fixed ANTsImage Mask for metric
        """
        if not isinstance(image, iio.ANTsImage):
            raise ValueError('image must be ANTsImage type')

        if image.dimension != self.dimension:
            raise ValueError('image dim (%i) does not match metric dim (%i)' % (image.dimension, self.dimension))

        self._metric.setMovingImage(image.pointer, True)

    def set_sampling(self, strategy='regular', percentage=1.):
        if strategy is None:
            strategy = 'regular'
        if percentage is None:
            percentage = 1.
        self._metric.setSampling(strategy, percentage)

    def initialize(self):
        self._metric.initialize()
        self._is_initialized = True

    def get_value(self):
        if not self._is_initialized:
            self.initialize()
        return self._metric.getValue()

    def __call__(self, fixed, moving, fixed_mask=None, moving_mask=None, sampling_strategy=None, sampling_percentage=None):
        self.set_fixed_image(fixed)
        self.set_moving_image(moving)

        if fixed_mask is not None:
            self.set_fixed_mask(fixed_mask)

        if moving_mask is not None:
            self.set_moving_mask(moving_mask)

        if (sampling_strategy is not None) or (sampling_percentage is not None):
            self.set_sampling(sampling_strategy, sampling_percentage)

        self.initialize()

        return self.get_value()

    def __repr__(self):
        s = "ANTsImageToImageMetric\n" +\
            '\t {:<10} : {}\n'.format('Dimension', self.dimension)+\
            '\t {:<10} : {}\n'.format('Precision', self.precision)+\
            '\t {:<10} : {}\n'.format('MetricType', self.metrictype)+\
            '\t {:<10} : {}\n'.format('IsVector', self.is_vector)
        return s


