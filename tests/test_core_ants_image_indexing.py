"""
Test ants_image.py

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


class TestClass_AntsImageIndexing(unittest.TestCase):

    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_pixeltype_2d(self):
        img = ants.image_read(ants.get_data('r16'))
        for ptype in ['unsigned char', 'unsigned int', 'float', 'double']:
            img = img.clone(ptype)
            self.assertEqual(img.pixeltype, ptype)
            img2 = img[:10,:10]
            self.assertEqual(img2.pixeltype, ptype)

    def test_pixeltype_3d(self):
        img = ants.image_read(ants.get_data('mni'))
        for ptype in ['unsigned char', 'unsigned int', 'float', 'double']:
            img = img.clone(ptype)
            self.assertEqual(img.pixeltype, ptype)
            img2 = img[:10,:10,:10]
            self.assertEqual(img2.pixeltype, ptype)
            img3 = img[:10,:10,10]
            self.assertEqual(img3.pixeltype, ptype)

    def test_2d(self):
        img = ants.image_read(ants.get_data('r16'))

        img2 = img[:10,:10]
        self.assertEqual(img2.dimension, 2)
        img2 = img[:5,:5]
        self.assertEqual(img2.dimension, 2)

        img2 = img[1:20,1:10]
        self.assertEqual(img2.dimension, 2)
        img2 = img[:5,:5]
        self.assertEqual(img2.dimension, 2)
        img2 = img[:5,4:5]
        self.assertEqual(img2.dimension, 2)

        img2 = img[5:5,5:5]

        # down to 1d
        arr = img[10,:]
        self.assertTrue(isinstance(arr, np.ndarray))

        # single value
        arr = img[10,10]

    def test_2d_image_index(self):
        img = ants.image_read(ants.get_data('r16'))
        idx = img > 200

        # acts like a mask
        img2 = img[idx]

    def test_3d(self):
        img = ants.image_read(ants.get_data('mni'))

        img2 = img[:10,:10,:10]
        self.assertEqual(img2.dimension, 3)
        img2 = img[:5,:5,:5]
        self.assertEqual(img2.dimension, 3)
        img2 = img[1:20,1:10,1:15]
        self.assertEqual(img2.dimension, 3)
        img2 = img[:5,:5,:5]
        self.assertEqual(img2.dimension, 3)

        # down to 2d
        img2 = img[10,:,:]
        self.assertEqual(img2.dimension, 2)
        img2 = img[:,10,:]
        self.assertEqual(img2.dimension, 2)
        img2 = img[:,:,10]
        self.assertEqual(img2.dimension, 2)
        img2 = img[10,:20,:20]
        self.assertEqual(img2.dimension, 2)
        img2 = img[:20,10,:]
        self.assertEqual(img2.dimension, 2)
        img2 = img[2:20,3:30,10]
        self.assertEqual(img2.dimension, 2)

        # down to 1d
        arr = img[10,:,10]
        self.assertTrue(isinstance(arr, np.ndarray))
        arr = img[10,:,5]
        self.assertTrue(isinstance(arr, np.ndarray))

        # single value
        arr = img[10,10,10]

    def test_double_indexing(self):
        img = ants.image_read(ants.get_data('mni'))
        img2 = img[20:,:,:]
        self.assertEqual(img2.shape, (162,218,182))

        img3 = img[0,:,:]
        self.assertEqual(img3.shape, (218,182))

    def test_reverse_error(self):
        img = ants.image_read(ants.get_data('mni'))
        with self.assertRaises(Exception):
            img2 = img[20:10,:,:]

    def test_2d_vector(self):
        img = ants.image_read(ants.get_data('r16'))
        img2 = img[:10,:10]

        img_v = ants.merge_channels([img])
        img_v2 = img_v[:10,:10]

        self.assertTrue(ants.allclose(img2, ants.split_channels(img_v2)[0]))

    def test_2d_vector_multi(self):
        img = ants.image_read(ants.get_data('r16'))
        img2 = img[:10,:10]

        img_v = ants.merge_channels([img,img,img])
        img_v2 = img_v[:10,:10]

        self.assertTrue(ants.allclose(img2, ants.split_channels(img_v2)[0]))
        self.assertTrue(ants.allclose(img2, ants.split_channels(img_v2)[1]))
        self.assertTrue(ants.allclose(img2, ants.split_channels(img_v2)[2]))

    def test_setting_3d(self):
        img = ants.image_read(ants.get_data('mni'))
        img2d = img[100,:,:]

        # setting a sub-image with an image
        img2 = img + 10
        img2[100,:,:] = img2d

        self.assertFalse(ants.allclose(img, img2))
        self.assertTrue(ants.allclose(img2d, img2[100,:,:]))

        # setting a sub-image with an array
        img2 = img + 10
        img2[100,:,:] = img2d.numpy()

        self.assertFalse(ants.allclose(img, img2))
        self.assertTrue(ants.allclose(img2d, img2[100,:,:]))

    def test_setting_2d(self):
        img = ants.image_read(ants.get_data('r16'))
        img2d = img[100,:]

        # setting a sub-image with an image
        img2 = img + 10
        img2[100,:] = img2d

        self.assertFalse(ants.allclose(img, img2))
        self.assertTrue(np.allclose(img2d, img2[100,:]))


    def test_setting_2d_sub_image(self):
        img = ants.image_read(ants.get_data('r16'))
        img2d = img[10:30,10:30]

        # setting a sub-image with an image
        img2 = img + 10
        img2[10:30,10:30] = img2d

        self.assertFalse(ants.allclose(img, img2))
        self.assertTrue(ants.allclose(img2d, img2[10:30,10:30]))

        # setting a sub-image with an array
        img2 = img + 10
        img2[10:30,10:30] = img2d.numpy()

        self.assertFalse(ants.allclose(img, img2))
        self.assertTrue(ants.allclose(img2d, img2[10:30,10:30]))

    def test_setting_correctness(self):

        img = ants.image_read(ants.get_data('r16')) * 0
        self.assertEqual(img.sum(), 0)

        img2 = img[10:30,10:30]
        img2 = img2 + 10
        self.assertEqual(img2.mean(), 10)

        img[:20,:20] = img2
        self.assertEqual(img.sum(), img2.sum())
        self.assertEqual(img.numpy()[:20,:20].sum(), img2.sum())
        self.assertNotEqual(img.numpy()[10:30,10:30].sum(), img2.sum())


    def test_slicing_3d(self):
        img = ants.image_read(ants.get_data('mni'))
        img2 = img[:10,:10,:10]
        img3 = img[10:20,10:20,10:20]

        self.assertTrue(ants.allclose(img2, img3))

        img_np = img.numpy()

        self.assertTrue(np.allclose(img2.numpy(), img_np[:10,:10,:10]))
        self.assertTrue(np.allclose(img[20].numpy(), img_np[20]))
        self.assertTrue(np.allclose(img[:,20:40].numpy(), img_np[:,20:40]))
        self.assertTrue(np.allclose(img[:,:,20:-2].numpy(), img_np[:,:,20:-2]))
        self.assertTrue(np.allclose(img[0:-1,].numpy(), img_np[0:-1,]))
        self.assertTrue(np.allclose(img[100,10:100,0:-1].numpy(), img_np[100,10:100,0:-1]))
        self.assertTrue(np.allclose(img[:,10:,30:].numpy(), img_np[:,10:,30:]))
        # if the slice returns 1D, it should be a numpy array already
        self.assertTrue(np.allclose(img[100:-1,30,40], img_np[100:-1,30,40]))

    def test_slicing_2d(self):
        img = ants.image_read(ants.get_data('r16'))

        img2 = img[:10,:10]

        img_np = img.numpy()

        self.assertTrue(np.allclose(img2.numpy(), img_np[:10,:10]))
        self.assertTrue(np.allclose(img[:,20:40].numpy(), img_np[:,20:40]))
        self.assertTrue(np.allclose(img[0:-1,].numpy(), img_np[0:-1,]))
        self.assertTrue(np.allclose(img[50:,10:-3].numpy(), img_np[50:,10:-3]))
        # if the slice returns 1D, it should be a numpy array already
        self.assertTrue(np.allclose(img[20], img_np[20]))
        self.assertTrue(np.allclose(img[100:-1,30], img_np[100:-1,30]))

if __name__ == '__main__':
    run_tests()