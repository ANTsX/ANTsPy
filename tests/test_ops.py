import os
import unittest
from common import run_tests

import ants

import numpy as np
import numpy.testing as nptest


class Test_slice_image(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_slice_image_2d(self):
        """
        Test that resampling an image doesnt cause the resampled
        image to have NaNs - previously caused by resampling an
        image of type DOUBLE
        """
        img = ants.image_read(ants.get_ants_data('r16'))
        
        img2 = ants.slice_image(img, 0, 100)
        self.assertTrue(isinstance(img2, np.ndarray))
        
        img2 = ants.slice_image(img, 1, 100)
        self.assertTrue(isinstance(img2, np.ndarray))
        
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, 2, 100)
        
    def test_slice_image_3d(self):
        """
        Test that resampling an image doesnt cause the resampled
        image to have NaNs - previously caused by resampling an
        image of type DOUBLE
        """
        img = ants.image_read(ants.get_ants_data('mni'))
        
        img2 = ants.slice_image(img, 0, 100)
        self.assertEqual(img2.dimension, 2)
        
        img2 = ants.slice_image(img, 1, 100)
        self.assertEqual(img2.dimension, 2)
        
        img2 = ants.slice_image(img, 2, 100)
        self.assertEqual(img2.dimension, 2)
        
        img2 = ants.slice_image(img.clone('unsigned int'), 2, 100)
        self.assertEqual(img2.dimension, 2)
        
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, 3, 100)
            
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, 2, 100, collapse_strategy=23)


if __name__ == '__main__':
    run_tests()