"""
Test ants_image.py

nptest.assert_allclose
self.assertEqual
self.assertTrue


"""

import os
import unittest
from common import run_tests
from context import ants
from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants


class TestModule_ants_image_io(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16')).clone('float')
        img3d = ants.image_read(ants.get_ants_data('mni')).clone('float')
        arr2d = np.random.randn(69,70).astype('float32')
        arr3d = np.random.randn(69,70,71).astype('float32')
        self.imgs = [img2d, img3d]
        self.arrs = [arr2d, arr3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_from_numpy(self):
        self.setUp()

        # no physical space info
        for arr in self.arrs:
            img = ants.from_numpy(arr)

            self.assertTrue(img.dimension, arr.ndim)
            self.assertTrue(img.shape, arr.shape)
            self.assertTrue(img.dtype, arr.dtype.name)
            nptest.assert_allclose(img.numpy(), arr)

            new_origin = tuple([6.9]*arr.ndim)
            new_spacing = tuple([3.6]*arr.ndim)
            new_direction = np.eye(arr.ndim)*9.6
            img2 = ants.from_numpy(arr, origin=new_origin, spacing=new_spacing, direction=new_direction)
            
            self.assertEqual(img2.origin, new_origin)
            self.assertEqual(img2.spacing, new_spacing)
            nptest.assert_allclose(img2.direction, new_direction)

        # test with components
        arr2d_components = np.random.randn(69,70,4).astype('float32')
        img = ants.from_numpy(arr2d_components, has_components=True)
        self.assertEqual(img.components, arr2d_components.shape[-1])
        nptest.assert_allclose(arr2d_components, img.numpy())

    def test_make_image(self):
        self.setUp()

        for arr in self.arrs:
            voxval = 6.
            img = ants.make_image(arr.shape, voxval=voxval)
            self.assertTrue(img.dimension, arr.ndim)
            self.assertTrue(img.shape, arr.shape)
            nptest.assert_allclose(img.mean(), voxval)

            new_origin = tuple([6.9]*arr.ndim)
            new_spacing = tuple([3.6]*arr.ndim)
            new_direction = np.eye(arr.ndim)*9.6
            img2 = ants.make_image(arr.shape, voxval=voxval, origin=new_origin, spacing=new_spacing, direction=new_direction)

            self.assertTrue(img2.dimension, arr.ndim)
            self.assertTrue(img2.shape, arr.shape)
            nptest.assert_allclose(img2.mean(), voxval)
            self.assertEqual(img2.origin, new_origin)
            self.assertEqual(img2.spacing, new_spacing)
            nptest.assert_allclose(img2.direction, new_direction)

            for ptype in self.pixeltypes:
                img = ants.make_image(arr.shape, voxval=1., pixeltype=ptype)
                self.assertEqual(img.pixeltype, ptype)

        # test with components
        img = ants.make_image((69,70,4), has_components=True)
        self.assertEqual(img.components, 4)
        self.assertEqual(img.dimension, 2)
        nptest.assert_allclose(img.mean(), 0.)

        img = ants.make_image((69,70,71,4), has_components=True)
        self.assertEqual(img.components, 4)
        self.assertEqual(img.dimension, 3)
        nptest.assert_allclose(img.mean(), 0.)




if __name__ == '__main__':
    run_tests()
