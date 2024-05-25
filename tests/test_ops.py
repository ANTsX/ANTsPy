import os
import unittest
from common import run_tests

import ants

import numpy as np
import numpy.testing as nptest


class Test_iMath(unittest.TestCase):
    
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_functions(self):
        image = ants.image_read(ants.get_data('r16'))
        ants.iMath_canny(image, sigma=2, lower=0, upper=1)
        ants.iMath_fill_holes(image, hole_type=2)
        ants.iMath_GC(image, radius=1)
        ants.iMath_GD(image, radius=1)
        ants.iMath_GE(image, radius=1)
        ants.iMath_GO(image, radius=1)
        ants.iMath_get_largest_component(image, min_size=50)
        ants.iMath_grad(image, sigma=0.5, normalize=False)
        ants.iMath_histogram_equalization(image, alpha=0.5, beta=0.5)
        ants.iMath_laplacian(image, sigma=0.5, normalize=False)
        ants.iMath_MC(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False)
        ants.iMath_MD(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False)
        ants.iMath_ME(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False)
        ants.iMath_MO(image, radius=1, value=1, shape=1, parametric=False, lines=3, thickness=1, include_center=False)
        ants.iMath_maurer_distance(image, foreground=1)
        ants.iMath_normalize(image)
        ants.iMath_pad(image, padding=2)
        ants.iMath_perona_malik(image, conductance=0.25, n_iterations=1)
        ants.iMath_sharpen(image)
        ants.iMath_truncate_intensity(image, lower_q=0.05, upper_q=0.95, n_bins=64)
    
    
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
            
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, -3, 100)
        
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, 4, 100)
            
        img2 = ants.slice_image(img, -1, 100)
        img3 = ants.slice_image(img, 1, 100)
        self.assertTrue(np.allclose(img2, img3))

    def test_slice_image_2d_vector(self):
        img0 = ants.image_read(ants.get_ants_data('r16'))
        img = ants.merge_channels([img0,img0,img0])

        img2 = ants.slice_image(img, 0, 100)
        self.assertTrue(isinstance(img2, np.ndarray))
        
        img2 = ants.slice_image(img, 1, 100)
        self.assertTrue(isinstance(img2, np.ndarray))
        
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, 2, 100)
            
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, -3, 100)
        
        with self.assertRaises(Exception):
            img2 = ants.slice_image(img, 4, 100)
            
        img2 = ants.slice_image(img, -1, 100)
        img3 = ants.slice_image(img, 1, 100)
        self.assertTrue(np.allclose(img2, img3))

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
            
        img2 = ants.slice_image(img, -1, 100)
        img3 = ants.slice_image(img, 2, 100)
        self.assertTrue(ants.allclose(img2, img3))

    def test_slice_image_3d_vector(self):
        """
        Test that resampling an image doesnt cause the resampled
        image to have NaNs - previously caused by resampling an
        image of type DOUBLE
        """
        img0 = ants.image_read(ants.get_ants_data('mni'))
        img = ants.merge_channels([img0,img0,img0])
        
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
            
        img2 = ants.slice_image(img, -1, 100)
        img3 = ants.slice_image(img, 2, 100)
        self.assertTrue(ants.allclose(img2, img3))


if __name__ == '__main__':
    run_tests()
