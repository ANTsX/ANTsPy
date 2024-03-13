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

# Prevent displaying figures
import matplotlib as mpl
backend_ =  mpl.get_backend() 
mpl.use("Agg")  

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
            ants.plot(img)
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
            ants.plot_ortho(img)
            ants.plot_ortho(img, filename=filename)

class TestModule_plot_ortho_stack(unittest.TestCase):

    def setUp(self):
        self.img = ants.image_read(ants.get_ants_data('mni'))

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        ants.plot_ortho_stack([self.img, self.img])
        ants.plot_ortho_stack([self.img, self.img], filename=filename)

class TestModule_plot_hist(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        for img in self.imgs:
            ants.plot_hist(img)

class TestModule_plot_grid(unittest.TestCase):

    def setUp(self):
        mni1 = ants.image_read(ants.get_data('mni'))
        mni2 = mni1.smooth_image(1.)
        mni3 = mni1.smooth_image(2.)
        mni4 = mni1.smooth_image(3.)
        self.images3d = np.asarray([[mni1, mni2],
                                    [mni3, mni4]])
        self.images2d = np.asarray([[mni1.slice_image(2,100), mni2.slice_image(2,100)],
                                    [mni3.slice_image(2,100), mni4.slice_image(2,100)]])

    def tearDown(self):
        pass

    def test_plot_example(self):
        ants.plot_grid(self.images3d, slices=100)
        # should take middle slices if none are given
        ants.plot_grid(self.images3d)
        # should work with 2d images
        ants.plot_grid(self.images2d)




    
if __name__ == '__main__':
    run_tests()
