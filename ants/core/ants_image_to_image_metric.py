"""
ANTs ImageToImageMetric class
"""


class ANTsImageToImageMetric(object):
    """
    ANTsImageToImageMetric class
    """

    def __init__(self, metric):
        self._metric = metric

    @property
    def precision(self):
        return self._metric.precision

    @property
    def dimension(self):
        return self._metric.dimension

    @property
    def pixeltype(self):
        return self._metric.precision

    @property
    def mtype(self):
        return self._metric.mtype

    @property
    def is_vector(self):
        return self._metric.isVector

    @property
    def pointer(self):
        return self._metric.pointer

    def __repr__(self):
        s = "ANTsImageToImageMetric\n" +\
            '\t {:<10} : {}\n'.format('Dimensions', self.dimension)+\
            '\t {:<10} : {}\n'.format('PixelType', self.pixeltype)+\
            '\t {:<10} : {}\n'.format('Type', self.type)+\
            '\t {:<10} : {}\n'.format('Is Vector', self.is_vector)
        return s


