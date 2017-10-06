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

class TestModule_surface(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_surf_example(self):
        ch2i = ants.image_read( ants.get_ants_data("ch2") )
        ch2seg = ants.threshold_image( ch2i, "Otsu", 3 )
        wm   = ants.threshold_image( ch2seg, 3, 3 )
        wm2 = wm.smooth_image( 1 ).threshold_image( 0.5, 1e15 )
        kimg = ants.weingarten_image_curvature( ch2i, 1.5  ).smooth_image( 1 )
        wmz = wm2.iMath("MD",3)
        rp = [(90,180,90), (90,180,270), (90,180,180)]
        filename = mktemp(suffix='.png')
        ants.surf( x=wm2, y=[kimg], z=[wmz],
                inflation_factor=255, overlay_limits=(-0.3,0.3), verbose = True,
                rotation_params = rp, filename=filename)

class TestModule_volume(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_vol_example(self):
        ch2i = ants.image_read( ants.get_ants_data("mni") )
        ch2seg = ants.threshold_image( ch2i, "Otsu", 3 )
        wm   = ants.threshold_image( ch2seg, 3, 3 )
        kimg = ants.weingarten_image_curvature( ch2i, 1.5  ).smooth_image( 1 )
        rp = [(90,180,90), (90,180,270), (90,180,180)]
        filename = mktemp(suffix='.png')
        result = ants.vol( wm, [kimg], quantlimits=(0.01,0.99), filename=filename)


class TestModule_render_surface_function(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_render_surface_function_example(self):
        mni = ants.image_read(ants.get_ants_data('mni'))
        mnia = ants.image_read(ants.get_ants_data('mnia'))
        filename = mktemp(suffix='.html')
        ants.render_surface_function(mni, mnia, alphasurf=0.1, auto_open=False, filename=filename)


class TestModule_plot(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_plot_example(self):
        for img in self.imgs:
            ants.plot(img)


class TestModule_create_tiled_mosaic(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_example(self):
        img = ants.image_read(ants.get_ants_data('ch2')).resample_image((3,3,3))
        p = ants.create_tiled_mosaic(img)

        # test with output
        outfile = mktemp(suffix='.png')
        p = ants.create_tiled_mosaic(img, output=outfile)

        # rgb is not none
        rgb = img.clone()
        p = ants.create_tiled_mosaic(img, rgb=rgb)

if __name__ == '__main__':
    run_tests()