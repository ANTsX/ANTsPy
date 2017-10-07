"""
Test ants.contrib module
"""


import os
import unittest

from common import run_tests

import numpy as np
import numpy.testing as nptest

import ants


class TestModule_transforms(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('ch2')).resample_image((2,2,2))
        self.imgs_2d = [img2d]
        self.imgs_3d = [img3d]

    def tearDown(self):
        pass

    # ---- Zoom ---- #

    def test_Zoom3D_example(self):
        img = ants.image_read(ants.get_data('ch2'))
        tx = ants.contrib.Zoom3D(zoom=(0.8,0.8,0.8))
        img2 = tx.transform(img)

    def test_Zoom3D(self):
        for img in self.imgs_3d:
            imgclone = img.clone()
            tx = ants.contrib.Zoom3D((0.8,0.8,0.8))
            img_zoom = tx.transform(img)

            # physical space shouldnt change .. ?
            self.assertTrue(ants.image_physical_space_consistency(img, img_zoom))
            # assert no unintended changes to passed-in img
            self.assertTrue(ants.image_physical_space_consistency(img, imgclone))
            self.assertTrue(ants.allclose(img, imgclone))

            # apply to cloned image to ensure deterministic nature
            img_zoom2 = tx.transform(imgclone)
            self.assertTrue(ants.image_physical_space_consistency(img_zoom, img_zoom2))
            self.assertTrue(ants.allclose(img_zoom,img_zoom2))

    def test_RandomZoom3D_example(self):
        img = ants.image_read(ants.get_data('ch2'))
        tx = ants.contrib.RandomZoom3D(zoom_range=(0.8,0.9))
        img2 = tx.transform(img)

    # ---- Rotation ---- #

    def test_Rotate3D_example(self):
        img = ants.image_read(ants.get_data('ch2'))
        tx = ants.contrib.Rotate3D(rotation=(10,-5,12))
        img2 = tx.transform(img)

    def test_Rotate3D(self):
        for img in self.imgs_3d:
            imgclone = img.clone()
            tx = ants.contrib.Rotate3D((10,-5,12))
            img_zoom = tx.transform(img)

            # physical space shouldnt change .. ?
            self.assertTrue(ants.image_physical_space_consistency(img, img_zoom))
            # assert no unintended changes to passed-in img
            self.assertTrue(ants.image_physical_space_consistency(img, imgclone))
            self.assertTrue(ants.allclose(img, imgclone))

            # apply to cloned image to ensure deterministic nature
            img_zoom2 = tx.transform(imgclone)
            self.assertTrue(ants.image_physical_space_consistency(img_zoom, img_zoom2))
            self.assertTrue(ants.allclose(img_zoom,img_zoom2))

    def test_RandomRotate3D_example(self):
        img = ants.image_read(ants.get_data('ch2'))
        tx = ants.contrib.RandomRotate3D(rotation_range=(-10,10))
        img2 = tx.transform(img)


if __name__ == '__main__':
    run_tests()

