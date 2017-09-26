"""
Test that any previously reported bugs are fixed

nptest.assert_allclose
self.assertEqual
self.assertTrue

"""

import os
import unittest
from common import run_tests

import ants

import numpy as np
import numpy.testing as nptest


class Test_bugs(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_resample_returns_NaNs(self):
        """
        Test that resampling an image doesnt cause the resampled
        image to have NaNs - previously caused by resampling an
        image of type DOUBLE
        """
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img2dr = ants.resample_image(img2d, (2,2), 0, 0)

        self.assertTrue(np.sum(np.isnan(img2dr.numpy())) == 0)

        img3d = ants.image_read(ants.get_ants_data('mni'))
        img3dr = ants.resample_image(img3d, (2,2,2), 0, 0)

        self.assertTrue(np.sum(np.isnan(img3dr.numpy())) == 0)


if __name__ == '__main__':
    run_tests()
