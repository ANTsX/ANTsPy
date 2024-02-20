"""
Test ants.learn module

nptest.assert_allclose
self.assertEqual
self.assertTrue
"""

import os
import unittest
from common import run_tests
from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants


class TestModule_plot(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        for img in self.imgs:
            ants.plot(img, filename=filename)


class TestModule_create_tiled_mosaic(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_example(self):
        img = ants.image_read(ants.get_ants_data('ch2')).resample_image((3,3,3))

        # test with output
        outfile = mktemp(suffix='.png')
        p = ants.create_tiled_mosaic(img, output=outfile)

        # rgb is not none
        rgb = img.clone()
        p = ants.create_tiled_mosaic(img, rgb=rgb, output=outfile)

if __name__ == '__main__':
    run_tests()