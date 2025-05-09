"""
Test ants.deeplearn module

nptest.assert_allclose
self.assertEqual
self.assertTrue
"""

import unittest
from common import run_tests

import numpy as np
import numpy.testing as nptest
import pandas as pd

import ants

class TestModule_crop_image_center(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        crop_size = (64, 67)
        image_cropped = ants.crop_image_center(image, crop_size)
        self.assertEqual(image_cropped.shape[0], crop_size[0])
        self.assertEqual(image_cropped.shape[1], crop_size[1])

class TestModule_pad_image_by_factor(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image_padded = ants.pad_image_by_factor(image, 3)
        self.assertEqual(image_padded.shape[0], 258)
        self.assertEqual(image_padded.shape[1], 258)

class TestModule_pad_or_crop_image_to_size(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        re_size = (64, 237)
        image_cropped = ants.pad_or_crop_image_to_size(image, re_size)
        self.assertEqual(image_cropped.shape[0], re_size[0])
        self.assertEqual(image_cropped.shape[1], re_size[1])

class TestModule_data_augmentation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image1 = ants.image_read(ants.get_ants_data("r16"))
        image2 = ants.image_read(ants.get_ants_data("r64"))
        data_aug = ants.data_augmentation([[image1], [image2]],
                                          number_of_simulations=5,
                                          reference_image=image1)
        self.assertEqual(len(data_aug['simulated_images']), 5)

class TestModule_randomly_transform_image_data(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image1 = ants.image_read(ants.get_ants_data("r16"))
        image2 = ants.image_read(ants.get_ants_data("r64"))
        data_aug = ants.randomly_transform_image_data(image1,
                                                      [[image1], [image2]],
                                                      number_of_simulations=5)
        self.assertEqual(len(data_aug['simulated_images']), 5)

class TestModule_regression_match_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image1 = ants.image_read(ants.get_ants_data("r16"))
        image2 = ants.image_read(ants.get_ants_data("r64"))
        image_matched = ants.regression_match_image(image1, image2)

class TestModule_extract_image_patches(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        patches = ants.extract_image_patches(image, (8, 16))

class TestModule_histogram_warp_image_intensities(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image_xfrm = ants.histogram_warp_image_intensities(image)

class TestModule_simulate_bias_field(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image_xfrm = ants.simulate_bias_field(image)

class TestModule_one_hot(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        seg = ants.kmeans_segmentation(image, 3)['segmentation']
        one_hot = ants.segmentation_to_one_hot(seg.numpy().astype('int'))        
        one_hot_inv = ants.one_hot_to_segmentation(one_hot, seg)        
        one_hot_c = ants.segmentation_to_one_hot(seg.numpy().astype('int'),
                                                 channel_first_ordering=True)        
        one_hot_c_inv = ants.one_hot_to_segmentation(one_hot_c, seg,
                                                     channel_first_ordering=True)        

if __name__ == "__main__":
    run_tests()
