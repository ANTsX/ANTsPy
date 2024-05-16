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
    
    def test_2d(self):
        img = ants.image_read(ants.get_data('r16'))
        
        img2 = img[:10,:10]
        img2[:5,:5]
        img2 = img[1:20,1:10]
        img2[:5,:5]
        img2[:5,4:5]
        
        img2[5:5,5:5]
        
        # down to 1d
        arr = img[10,:]
        
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
        img2[:5,:5,:5]
        img2 = img[1:20,1:10,1:15]
        img2[:5,:5,:5]
        
        # down to 2d
        img2 = img[10,:,:]
        img2 = img[:,10,:]
        img2 = img[:,:,10]
        img2 = img[10,:20,:20]
        img2 = img[:20,10,:]
        img2 = img[2:20,3:30,10]
        
        # down to 1d
        arr = img[10,:,10]
        arr = img[10,:,5]
        
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
        

if __name__ == '__main__':
    run_tests()
