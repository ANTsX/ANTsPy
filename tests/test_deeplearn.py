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



class TestImagePatchesAndAugmentation(unittest.TestCase):
    def test_extract_reconstruct_roundtrip_scalar_and_multicomponent(self):
        for dim in (2, 3):
            shape = (9,) * dim
            data = np.arange(np.prod(shape), dtype='float32').reshape(shape)
            for components in (False, True):
                values = np.stack((data, data + 100), axis=-1) if components else data
                image = ants.from_numpy(values, has_components=components)
                for stride in (1, 2):
                    for as_array in (False, True):
                        patches = ants.extract_image_patches(image, (3,) * dim, stride_length=stride, return_as_array=as_array)
                        result = ants.reconstruct_image_from_patches(patches, image, stride_length=stride)
                        np.testing.assert_allclose(result.numpy(), values)

    def test_masked_patches_include_all_valid_edges(self):
        for dim in (2, 3):
            shape = (9,) * dim
            data = np.arange(np.prod(shape), dtype='float32').reshape(shape)
            image = ants.from_numpy(data)
            mask = image * 0 + 1
            for size in (2, 3, 4, 5, 6, 7):
                patches = ants.extract_image_patches(image, (size,) * dim, max_number_of_patches=10000, mask_image=mask, randomize=False)
                self.assertEqual(len(patches), (10 - size) ** dim)
                self.assertTrue(all(p.shape == (size,) * dim for p in patches))
                result = ants.reconstruct_image_from_patches(patches, mask, domain_image_is_mask=True)
                np.testing.assert_allclose(result.numpy(), data)

    def test_odd_patches_center_on_sparse_mask_voxels(self):
        image = ants.from_numpy(np.arange(81, dtype='float32').reshape(9, 9))
        for size in (3, 5, 7):
            offset = (size - 1) // 2
            mask_data = np.zeros((9, 9), dtype='float32')
            mask_data[offset, offset] = 1
            mask_data[8 - offset, 8 - offset] = 1
            mask = ants.from_numpy(mask_data)
            patches = ants.extract_image_patches(image, (size, size), max_number_of_patches=10, mask_image=mask, randomize=False)
            self.assertEqual(len(patches), 2)
            np.testing.assert_array_equal(patches[0], image.numpy()[:size, :size])
            np.testing.assert_array_equal(patches[1], image.numpy()[-size:, -size:])

    def test_crop_from_physical_center_with_rotated_direction(self):
        data = np.arange(400, dtype='float32').reshape(20, 20)
        image = ants.from_numpy(data, origin=(3, 8), spacing=(2, 3), direction=np.array([[0., -1.], [1., 0.]]))
        center = ants.transform_index_to_physical_point(image, (9, 9))
        result = ants.crop_image_from_center_point(image, center, (3, 3))
        np.testing.assert_allclose(result.numpy(), data[8:11, 8:11])

    def test_random_transforms_preserve_input_image_lists(self):
        reference = ants.from_numpy(np.ones((8, 8), dtype='float32'))
        moving = ants.from_numpy(np.ones((6, 6), dtype='float32'))
        images = [[moving, moving]]
        result = ants.randomly_transform_image_data(reference, images, number_of_simulations=3, transform_type='translation', sd_affine=0)
        self.assertEqual(len(images[0]), 2)
        self.assertIs(images[0][0], moving)
        self.assertTrue(all(len(sample) == 2 for sample in result['simulated_images']))

if __name__ == "__main__":
    run_tests()
