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

class TestModule_plot_ortho(unittest.TestCase):

    def setUp(self):
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img3d]

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        for img in self.imgs:
            ants.plot_ortho(img, filename=filename)

class TestModule_plot_ortho_stack(unittest.TestCase):

    def setUp(self):
        self.img = ants.image_read(ants.get_ants_data('mni'))

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        ants.plot_ortho_stack([self.img, self.img], filename=filename)

if __name__ == '__main__':
    run_tests()