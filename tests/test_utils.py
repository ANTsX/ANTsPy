"""
Test utils module
"""

import os
import unittest

from common import run_tests

import numpy.testing as nptest
import numpy as np
import pandas as pd

import ants


class TestModule_iMath(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        #img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d]

    def tearDown(self):
        pass

    def test_iMath_example(self):
        img = ants.image_read(ants.get_ants_data('r16'))
        img2 = ants.iMath(img, 'Canny', 1, 5, 12)

    def test_iMath_options(self):
        for img in self.imgs:
            # Grayscale dilation
            img_gd = ants.iMath(img, 'GD', 3)
            self.assertTrue(ants.image_physical_space_consistency(img_gd, img))
            self.assertTrue(not ants.allclose(img_gd, img))

            # Grayscale erosion
            img_tx = ants.iMath(img, 'GE', 3)
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertFalse(ants.allclose(img_tx, img))

            # Morphological dilation
            img_tx = ants.iMath(img, 'MD', 3)
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertFalse(ants.allclose(img_tx, img))

            # Morphological erosion
            img_tx = ants.iMath(img, 'ME', 3)
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertFalse(ants.allclose(img_tx, img))

            # Morphological closing
            img_tx = ants.iMath(img, 'MC', 3)
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertTrue(not ants.allclose(img_tx, img))

            # Morphological opening
            img_tx = ants.iMath(img, 'MO', 3)
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertTrue(not ants.allclose(img_tx, img))

            # Pad image - ERRORS
            #img_tx = ants.iMath(img, 'PadImage', 2)

            # PeronaMalik - ERRORS
            #img_tx = ants.iMath(img, 'PeronaMalik', 10, 0.5)

            # Maurer Distance
            img_tx = ants.iMath(img, 'MaurerDistance')
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertTrue(not ants.allclose(img_tx, img))

            # Danielsson Distance Map
            #img_tx = ants.iMath(img, 'DistanceMap')
            #self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            #self.assertTrue(not ants.allclose(img_tx, img))

            # Image gradient
            img_tx = ants.iMath(img, 'Grad', 1)
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertTrue(not ants.allclose(img_tx, img))

            # Laplacian
            img_tx = ants.iMath(img, 'Laplacian', 1, 1)
            self.assertTrue(ants.image_physical_space_consistency(img_tx, img))
            self.assertTrue(not ants.allclose(img_tx, img))

class TestModule_bias_correction(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_n3_bias_field_correction_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image_n3 = ants.n3_bias_field_correction(image)

    def test_n3_bias_field_correction(self):
        for img in self.imgs:
            image_n3 = ants.n3_bias_field_correction(img)

    def test_n4_bias_field_correction_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image_n4 = ants.n4_bias_field_correction(image)

        # spline param list
        image_n4 = ants.n4_bias_field_correction(image, spline_param=(10, 10))

        # image not float
        image_ui = image.clone("unsigned int")
        image_n4 = ants.n4_bias_field_correction(image_ui)

        # weight mask
        mask = ants.image_clone(image > image.mean(), pixeltype="float")
        ants.n4_bias_field_correction(image, weight_mask=mask)

        # len(spline_param) != img.dimension
        with self.assertRaises(Exception):
            ants.n4_bias_field_correction(image, spline_param=(10, 10, 10))

        # weight mask not ANTsImage
        with self.assertRaises(Exception):
            ants.n4_bias_field_correction(image, weight_mask=0.4)

    def test_n4_bias_field_correction(self):
        # def n4_bias_field_correction(image, mask=None, shrink_factor=4,
        #                     convergence={'iters':[50,50,50,50], 'tol':1e-07},
        #                     spline_param=200, verbose=False, weight_mask=None):
        for img in self.imgs:
            image_n4 = ants.n4_bias_field_correction(img)
            mask = ants.image_clone(img > img.mean(), pixeltype="float")
            image_n4 = ants.n4_bias_field_correction(img, mask=mask)

    def test_abp_n4_example(self):
        img = ants.image_read(ants.get_ants_data("r16"))
        img = ants.iMath(img, "Normalize") * 255.0
        img2 = ants.abp_n4(img)

    def test_abp_n4(self):
        # def abp_n4(image, intensity_truncation=(0.025,0.975,256), mask=None, usen3=False):
        for img in self.imgs:
            img2 = ants.abp_n4(img)
            img3 = ants.abp_n4(img, usen3=True)

        # intensity trunction doesnt have three values
        with self.assertRaises(Exception):
            img = self.imgs[0]
            ants.abp_n4(img, intensity_truncation=(1, 2))


class TestModule_channels(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni"))
        self.imgs = [img2d, img3d]
        arr2d = np.random.randn(69, 70, 4).astype("float32")
        arr3d = np.random.randn(69, 70, 71, 4).astype("float32")
        self.comparrs = [arr2d, arr3d]

    def tearDown(self):
        pass

    def test_merge_channels(self):
        for img in self.imgs:
            imglist = [img.clone(), img.clone()]
            merged_img = ants.merge_channels(imglist)

            self.assertEqual(merged_img.shape, img.shape)
            self.assertEqual(merged_img.components, len(imglist))
            nptest.assert_allclose(merged_img.numpy()[..., 0], imglist[0].numpy())

            imglist = [img.clone(), img.clone(), img.clone(), img.clone()]
            merged_img = ants.merge_channels(imglist)

            self.assertEqual(merged_img.shape, img.shape)
            self.assertEqual(merged_img.components, len(imglist))
            nptest.assert_allclose(merged_img.numpy()[..., 0], imglist[-1].numpy())

        for comparr in self.comparrs:
            imglist = [
                ants.from_numpy(comparr[..., i].copy())
                for i in range(comparr.shape[-1])
            ]
            merged_img = ants.merge_channels(imglist)
            for i in range(comparr.shape[-1]):
                nptest.assert_allclose(merged_img.numpy()[..., i], comparr[..., i])


class TestModule_crop_image(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_crop_image_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        cropped = ants.crop_image(fi)
        cropped = ants.crop_image(fi, fi, 100)

        # image not float type
        cropped = ants.crop_image(fi.clone("unsigned int"))

        # label image not float
        cropped = ants.crop_image(fi, fi.clone("unsigned int"), 100)

        # channel image
        fi = ants.image_read( ants.get_ants_data('r16') )
        cropped = ants.crop_image(fi)
        fi2 = ants.merge_channels([fi,fi])
        cropped2 = ants.crop_image(fi2)

        self.assertEqual(cropped.shape, cropped2.shape)

    def test_crop_indices_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        cropped = ants.crop_indices(fi, (10, 10), (100, 100))
        cropped = ants.smooth_image(cropped, 5)
        decropped = ants.decrop_image(cropped, fi)

        # image not float
        cropped = ants.crop_indices(fi.clone("unsigned int"), (10, 10), (100, 100))

        # image dim not equal to indices
        with self.assertRaises(Exception):
            cropped = ants.crop_indices(fi, (10, 10, 10), (100, 100))
            cropped = ants.crop_indices(fi, (10, 10), (100, 100, 100))

        # vector images
        fi = ants.image_read( ants.get_ants_data("r16"))
        cropped = ants.crop_indices( fi, (10,10), (100,100) )
        fi2 = ants.merge_channels([fi,fi])
        cropped2 = ants.crop_indices( fi, (10,10), (100,100) )
        self.assertEqual(cropped.shape, cropped2.shape)

    def test_decrop_image_example(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        mask = ants.get_mask(fi)
        cropped = ants.crop_image(fi, mask, 1)
        cropped = ants.smooth_image(cropped, 1)
        decropped = ants.decrop_image(cropped, fi)

        # image not float
        cropped = ants.crop_image(fi.clone("unsigned int"), mask, 1)

        # full image not float
        cropped = ants.crop_image(fi, mask.clone("unsigned int"), 1)

class TestModule_pad_image(unittest.TestCase):
    def setUp(self):
        self.img2d = ants.image_read(ants.get_ants_data("r16"))
        self.img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))

    def tearDown(self):
        pass

    def test_pad_image_example(self):
        img = self.img2d.clone()
        img.set_origin((0, 0))
        img.set_spacing((2, 2))
        # isotropic pad via pad_width
        padded = ants.pad_image(img, pad_width=(10, 10))
        self.assertEqual(padded.shape, (img.shape[0] + 10, img.shape[1] + 10))
        for img_orig_elem, pad_orig_elem in zip(img.origin, padded.origin):
            self.assertAlmostEqual(pad_orig_elem, img_orig_elem - 10, places=3)

        img = self.img2d.clone()
        img.set_origin((0, 0))
        img.set_spacing((2, 2))
        img = ants.resample_image(img, (128,128), 1, 1)
        # isotropic pad via shape
        padded = ants.pad_image(img, shape=(160,160))
        self.assertSequenceEqual(padded.shape, (160, 160))

        img = self.img3d.clone()
        img = ants.resample_image(img, (128,160,96), 1, 1)
        padded = ants.pad_image(img)
        self.assertSequenceEqual(padded.shape, (160, 160, 160))

        # pad only on superior side
        img = self.img3d.clone()
        padded = ants.pad_image(img, pad_width=[(0,4),(0,8),(0,12)])
        self.assertSequenceEqual(padded.shape, (img.shape[0] + 4, img.shape[1] + 8, img.shape[2] + 12))
        self.assertSequenceEqual(img.origin, padded.origin)


class TestModule_denoise_image(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_denoise_image_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        # add fairly large salt and pepper noise
        imagenoise = image + np.random.randn(*image.shape).astype("float32") * 5
        imagedenoise = ants.denoise_image(imagenoise, image > image.mean())


class TestModule_fit_bspline_object_to_scattered_data(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fit_bspline_object_to_scattered_data_1d_curve_example(self):
        x = np.linspace(-4, 4, num=100) + np.random.uniform(-0.1, 0.1, 100)
        u = np.linspace(0, 1.0, num=len(x))
        scattered_data = np.expand_dims(u, axis=-1)
        parametric_data = np.expand_dims(u, axis=-1)
        spacing = 1/(len(x)-1) * 1.0
        bspline_curve = ants.fit_bspline_object_to_scattered_data(
            scattered_data, parametric_data,
            parametric_domain_origin=[0.0], parametric_domain_spacing=[spacing],
            parametric_domain_size=[len(x)], is_parametric_dimension_closed=None,
            number_of_fitting_levels=5, mesh_size=1)

    def test_fit_bspline_object_to_scattered_data_2d_curve_example(self):
        x = np.linspace(-4, 4, num=100)
        y = np.exp(-np.multiply(x, x)) + np.random.uniform(-0.1, 0.1, len(x))
        u = np.linspace(0, 1.0, num=len(x))
        scattered_data = np.column_stack((x, y))
        parametric_data = np.expand_dims(u, axis=-1)
        spacing = 1/(len(x)-1) * 1.0
        bspline_curve = ants.fit_bspline_object_to_scattered_data(
            scattered_data, parametric_data,
            parametric_domain_origin=[0.0], parametric_domain_spacing=[spacing],
            parametric_domain_size=[len(x)], is_parametric_dimension_closed=None,
            number_of_fitting_levels=5, mesh_size=1)

    def test_fit_bspline_object_to_scattered_data_3d_curve_example(self):
        x = np.linspace(-4, 4, num=100)
        y = np.exp(-np.multiply(x, x)) + np.random.uniform(-0.1, 0.1, len(x))
        z = np.exp(-np.multiply(x, y)) + np.random.uniform(-0.1, 0.1, len(x))
        u = np.linspace(0, 1.0, num=len(x))
        scattered_data = np.column_stack((x, y))
        parametric_data = np.expand_dims(u, axis=-1)
        spacing = 1/(len(x)-1) * 1.0
        bspline_curve = ants.fit_bspline_object_to_scattered_data(
            scattered_data, parametric_data,
            parametric_domain_origin=[0.0], parametric_domain_spacing=[spacing],
            parametric_domain_size=[len(x)], is_parametric_dimension_closed=None,
            number_of_fitting_levels=5, mesh_size=1)

    def test_fit_bspline_object_to_scattered_data_2d_scalar_field_example(self):
        number_of_random_points = 10000
        img = ants.image_read( ants.get_ants_data("r16"))
        img_array = img.numpy()
        row_indices = np.random.choice(range(2, img_array.shape[0]), number_of_random_points)
        col_indices = np.random.choice(range(2, img_array.shape[1]), number_of_random_points)
        scattered_data = np.zeros((number_of_random_points, 1))
        parametric_data = np.zeros((number_of_random_points, 2))
        for i in range(number_of_random_points):
            scattered_data[i,0] = img_array[row_indices[i], col_indices[i]]
            parametric_data[i,0] = row_indices[i]
            parametric_data[i,1] = col_indices[i]
        bspline_img = ants.fit_bspline_object_to_scattered_data(
            scattered_data, parametric_data,
            parametric_domain_origin=[0.0, 0.0],
            parametric_domain_spacing=[1.0, 1.0],
            parametric_domain_size = img.shape,
            number_of_fitting_levels=7, mesh_size=1)

    def test_fit_bspline_object_to_scattered_data_3d_scalar_field_example(self):
        number_of_random_points = 10000
        img = ants.image_read( ants.get_ants_data("mni"))
        img_array = img.numpy()
        row_indices = np.random.choice(range(2, img_array.shape[0]), number_of_random_points)
        col_indices = np.random.choice(range(2, img_array.shape[1]), number_of_random_points)
        dep_indices = np.random.choice(range(2, img_array.shape[2]), number_of_random_points)
        scattered_data = np.zeros((number_of_random_points, 1))
        parametric_data = np.zeros((number_of_random_points, 3))
        for i in range(number_of_random_points):
            scattered_data[i,0] = img_array[row_indices[i], col_indices[i], dep_indices[i]]
            parametric_data[i,0] = row_indices[i]
            parametric_data[i,1] = col_indices[i]
            parametric_data[i,2] = dep_indices[i]
        bspline_img = ants.fit_bspline_object_to_scattered_data(
            scattered_data, parametric_data,
            parametric_domain_origin=[0.0, 0.0, 0.0],
            parametric_domain_spacing=[1.0, 1.0, 1.0],
            parametric_domain_size = img.shape,
            number_of_fitting_levels=5, mesh_size=1)

    def test_fit_bspline_object_to_scattered_data_2d_displacement_field_example(self):
            img = ants.image_read( ants.get_ants_data("r16"))
            # smooth a single vector in the middle of the image
            scattered_data = np.reshape(np.asarray((10.0, 10.0)), (1, 2))
            parametric_data = np.reshape(np.asarray(
                (scattered_data[0, 0]/img.shape[0], scattered_data[0, 1]/img.shape[1])), (1, 2))
            bspline_field = ants.fit_bspline_object_to_scattered_data(
                scattered_data, parametric_data,
                parametric_domain_origin=[0.0, 0.0],
                parametric_domain_spacing=[1.0, 1.0],
                parametric_domain_size = img.shape,
                number_of_fitting_levels=7, mesh_size=1)

    def test_fit_bspline_object_to_scattered_data_3d_displacement_field_example(self):
            img = ants.image_read( ants.get_ants_data("mni"))
            # smooth a single vector in the middle of the image
            scattered_data = np.reshape(np.asarray((10.0, 10.0, 10.0)), (1, 3))
            parametric_data = np.reshape(np.asarray(
                (scattered_data[0, 0]/img.shape[0],
                 scattered_data[0, 1]/img.shape[1],
                 scattered_data[0, 2]/img.shape[2])), (1, 3))
            bspline_field = ants.fit_bspline_object_to_scattered_data(
                scattered_data, parametric_data,
                parametric_domain_origin=[0.0, 0.0, 0.0],
                parametric_domain_spacing=[1.0, 1.0, 1.0],
                parametric_domain_size = img.shape,
                number_of_fitting_levels=5, mesh_size=1)


class TestModule_get_ants_data(unittest.TestCase):
    def test_get_ants_data(self):
        for dataname in ants.get_ants_data(None):
            if dataname.endswith("nii.gz"):
                img = ants.image_read(dataname)

#    def test_get_ants_data_files(self):
#        datanames = ants.get_ants_data(None)
#        self.assertTrue(isinstance(datanames, list))
#        self.assertTrue(len(datanames) > 0)

    def test_get_ants_data2(self):
        for dataname in ants.get_data(None):
            if dataname.endswith("nii.gz"):
                img = ants.image_read(dataname)

#    def test_get_ants_data_files2(self):
#        datanames = ants.get_data(None)
#        self.assertTrue(isinstance(datanames, list))
#        self.assertTrue(len(datanames) > 0)


class TestModule_get_centroids(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_get_centroids_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image = ants.threshold_image(image, 90, 120)
        image = ants.label_clusters(image, 10)
        cents = ants.get_centroids(image)


class TestModule_get_mask(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_get_mask_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        mask = ants.get_mask(image)


class TestModule_get_neighborhood(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_get_neighborhood_in_mask_example(self):
        img = ants.image_read(ants.get_ants_data("r16"))
        mask = ants.get_mask(img)
        mat = ants.get_neighborhood_in_mask(img, mask, radius=(2, 2))

        # image not ANTsImage
        with self.assertRaises(Exception):
            ants.get_neighborhood_in_mask(2, mask, radius=(2, 2))
        # mask not ANTsImage
        with self.assertRaises(Exception):
            ants.get_neighborhood_in_mask(img, 2, radius=(2, 2))
        # radius not right length
        with self.assertRaises(Exception):
            ants.get_neighborhood_in_mask(img, 2, radius=(2, 2, 2, 2))

        # radius is just a float/int
        mat = ants.get_neighborhood_in_mask(img, mask, radius=2)

        # boundary condition == 'image'
        mat = ants.get_neighborhood_in_mask(
            img, mask, radius=(2, 2), boundary_condition="image"
        )
        # boundary condition == 'mean'
        mat = ants.get_neighborhood_in_mask(
            img, mask, radius=(2, 2), boundary_condition="mean"
        )

        # spatial info
        mat = ants.get_neighborhood_in_mask(img, mask, radius=(2, 2), spatial_info=True)

        # get_gradient
        mat = ants.get_neighborhood_in_mask(img, mask, radius=(2, 2), get_gradient=True)

    def test_get_neighborhood_at_voxel_example(self):
        img = ants.image_read(ants.get_ants_data("r16"))
        center = (2, 2)
        radius = (3, 3)
        retval = ants.get_neighborhood_at_voxel(img, center, radius)

        # image not ANTsImage
        with self.assertRaises(Exception):
            ants.get_neighborhood_at_voxel(2, center, radius)
        # wrong center length
        with self.assertRaises(Exception):
            ants.get_neighborhood_at_voxel(img, (2, 2, 2, 2, 2), radius)
        # wrong radius length
        with self.assertRaises(Exception):
            ants.get_neighborhood_at_voxel(img, center, (2, 2, 2, 2))


class TestModule_image_similarity(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_image_similarity_example(self):
        x = ants.image_read(ants.get_ants_data("r16"))
        y = ants.image_read(ants.get_ants_data("r30"))
        metric = ants.image_similarity(x, y, metric_type="MeanSquares")
        self.assertTrue(metric > 0)


class TestModule_image_to_cluster_images(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data("r16"))
        img3d = ants.image_read(ants.get_ants_data("mni")).resample_image((2, 2, 2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_image_to_cluster_images_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image = ants.threshold_image(image, 1, 1e15)
        image_cluster_list = ants.image_to_cluster_images(image)


class TestModule_label_clusters(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_label_clusters_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        timageFully = ants.label_clusters(image, 10, 128, 150, True)
        timageFace = ants.label_clusters(image, 10, 128, 150, False)


class TestModule_label_image_centroids(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_label_image_centroids(self):
        image = ants.from_numpy(
            np.asarray([[[0, 2], [1, 3]], [[4, 6], [5, 7]]]).astype("float32")
        )
        labels = ants.label_image_centroids(image)
        self.assertEqual(len(labels['labels']), 7)

        # Test non-sequential labels
        image = ants.from_numpy(
            np.asarray([[[0, 2], [2, 2]], [[2, 0], [5, 0]]]).astype("float32")
        )

        labels = ants.label_image_centroids(image)
        self.assertTrue(len(labels['labels']) == 2)
        self.assertTrue(labels['labels'][1] == 5)
        self.assertTrue(np.allclose(labels['vertices'][0], [0.5 , 0.5 , 0.25], atol=1e-5))
        # With convex = False, the centroid position should change
        labels = ants.label_image_centroids(image, convex=False)
        self.assertTrue(np.allclose(labels['vertices'][0], [1.0, 1.0, 0.0], atol=1e-5))
        # single point unchanged
        self.assertTrue(np.allclose(labels['vertices'][1], [0.0, 1.0, 1.0], atol=1e-5))


class TestModule_label_overlap_measures(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_label_overlap_measures(self):
        r16 = ants.image_read( ants.get_ants_data('r16') )
        r64 = ants.image_read( ants.get_ants_data('r64') )
        s16 = ants.kmeans_segmentation( r16, 3 )['segmentation']
        s64 = ants.kmeans_segmentation( r64, 3 )['segmentation']
        stats = ants.label_overlap_measures(s16, s64)


class TestModule_label_stats(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_label_stats_example(self):
        image = ants.image_read(ants.get_ants_data("r16"), 2)
        image = ants.resample_image(image, (64, 64), 1, 0)
        segs1 = ants.kmeans_segmentation(image, 3)
        stats = ants.label_stats(image, segs1["segmentation"])


class TestModule_labels_to_matrix(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_labels_to_matrix_example(self):
        fi = ants.image_read(ants.get_ants_data("r16")).resample_image((60, 60), 1, 0)
        mask = ants.get_mask(fi)
        labs = ants.kmeans_segmentation(fi, 3)["segmentation"]
        labmat = ants.labels_to_matrix(labs, mask)


class TestModule_make_points_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_make_points_image_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        points = np.array([[102,  76],[134, 129]])
        points_image = ants.make_points_image(points, image, radius=5)
        stats = ants.label_stats(image, points_image)
        self.assertTrue(np.allclose(stats['Volume'].to_numpy()[1:3], 97.0, atol=1e-5))
        self.assertTrue(np.allclose(stats['x'].to_numpy()[1:3], points[:,0], atol=1e-5))
        self.assertTrue(np.allclose(stats['y'].to_numpy()[1:3], points[:,1], atol=1e-5))

        points = np.array([[102,  76,  50],[134, 129,  50]])
        # Shouldn't allow 3D points on a 2D image
        with self.assertRaises(Exception):
            points_image = ants.make_points_image(image, points, radius=3)


class TestModule_mask_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mask_image_example(self):
        myimage = ants.image_read(ants.get_ants_data("r16"))
        mask = ants.get_mask(myimage)
        myimage_mask = ants.mask_image(myimage, mask, (2, 3))
        seg = ants.kmeans_segmentation(myimage, 3)
        myimage_mask = ants.mask_image(myimage, seg["segmentation"], (1, 3))


class TestModule_mni2tal(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mni2tal_example(self):
        ants.mni2tal((10, 12, 14))


class TestModule_morphology(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_morphology(self):
        fi = ants.image_read(ants.get_ants_data("r16"))
        mask = ants.get_mask(fi)
        dilated_ball = ants.morphology(
            mask, operation="dilate", radius=3, mtype="binary", shape="ball"
        )
        eroded_box = ants.morphology(
            mask, operation="erode", radius=3, mtype="binary", shape="box"
        )
        opened_annulus = ants.morphology(
            mask,
            operation="open",
            radius=5,
            mtype="binary",
            shape="annulus",
            thickness=2,
        )

        ## --- ##
        mtype = "binary"
        mask = mask.clone()
        for operation in {"dilate", "erode", "open", "close"}:
            for shape in {"ball", "box", "cross", "annulus", "polygon"}:
                morph = ants.morphology(
                    mask, operation=operation, radius=3, mtype=mtype, shape=shape
                )
                self.assertTrue(isinstance(morph, ants.core.ants_image.ANTsImage))
        # invalid operation
        with self.assertRaises(Exception):
            ants.morphology(
                mask, operation="invalid-operation", radius=3, mtype=mtype, shape=shape
            )

        mtype = "grayscale"
        img = ants.image_read(ants.get_ants_data("r16"))
        for operation in {"dilate", "erode", "open", "close"}:
            morph = ants.morphology(img, operation=operation, radius=3, mtype=mtype)
            self.assertTrue(isinstance(morph, ants.core.ants_image.ANTsImage))
        # invalid operation
        with self.assertRaises(Exception):
            ants.morphology(
                img, operation="invalid-operation", radius=3, mtype="grayscale"
            )

        # invalid morphology type
        with self.assertRaises(Exception):
            ants.morphology(
                img, operation="dilate", radius=3, mtype="invalid-morphology"
            )

        # invalid shape
        with self.assertRaises(Exception):
            ants.morphology(
                mask,
                operation="dilate",
                radius=3,
                mtype="binary",
                shape="invalid-shape",
            )


class TestModule_ndimage_to_list(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ndimage_to_list(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        image2 = ants.image_read(ants.get_ants_data("r64"))
        ants.set_spacing(image, (2, 2))
        ants.set_spacing(image2, (2, 2))
        imageTar = ants.make_image((*image2.shape, 2))
        ants.set_spacing(imageTar, (2, 2, 2))
        image3 = ants.list_to_ndimage(imageTar, [image, image2])
        self.assertEqual(image3.dimension, 3)
        ants.set_direction(image3, np.eye(3) * 2)
        images_unmerged = ants.ndimage_to_list(image3)
        self.assertEqual(len(images_unmerged), 2)
        self.assertEqual(images_unmerged[0].dimension, 2)


class TestModule_simulate_displacement_field(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simulate_exponential_displacement_field_2d(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        exp_field = ants.simulate_displacement_field(image, field_type="exponential")

    def test_simulate_bspline_displacement_field_2d(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        bsp_field = ants.simulate_displacement_field(image, field_type="bspline")


class TestModule_smooth_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_smooth_image_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        simage = ants.smooth_image(image, (1.2, 1.5))


class TestModule_threshold_image(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_threshold_image_example(self):
        image = ants.image_read(ants.get_ants_data("r16"))
        timage = ants.threshold_image(image, 0.5, 1e15)


class TestModule_weingarten_image_curvature(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_weingarten_image_curvature_example(self):
        image = ants.image_read(ants.get_ants_data("mni")).resample_image((3, 3, 3))
        imagecurv = ants.weingarten_image_curvature(image)

        image2 = ants.image_read(ants.get_ants_data("r16")).resample_image((2, 2))
        imagecurv2 = ants.weingarten_image_curvature(image2)


class TestModule_scalar_rgb_vector(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        import ants
        import numpy as np

        # this fails because ConvertScalarImageToRGB is not wrapped
        img = ants.image_read(ants.get_data("r16"), pixeltype="unsigned char")
        # img_rgb = img.scalar_to_rgb()
        # img_vec = img_rgb.rgb_to_vector()

        # rgb_arr = img_rgb.numpy()
        # vec_arr = img_vec.numpy()
        print(np.allclose(img.numpy(), img.numpy()))

    def test2(self):
        import ants
        import numpy as np

        rgb_img = ants.from_numpy(
            np.random.randint(0, 255, (20, 20, 3)).astype("uint8"), is_rgb=True
        )
        vec_img = rgb_img.rgb_to_vector()
        print(ants.allclose(rgb_img, vec_img))

        vec_img = ants.from_numpy(
            np.random.randint(0, 255, (20, 20, 3)).astype("uint8"), has_components=True
        )
        rgb_img = vec_img.vector_to_rgb()
        print(ants.allclose(rgb_img, vec_img))


class TestRandom(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_bspline_field(self):
        points = np.array([[-50, -50]])
        deltas = np.array([[10, 10]])
        bspline_field = ants.fit_bspline_displacement_field(
        displacement_origins=points, displacements=deltas,
        origin=[0.0, 0.0], spacing=[1.0, 1.0], size=[100, 100],
        direction=np.array([[-1, 0], [0, -1]]),
        number_of_fitting_levels=4, mesh_size=(1, 1))

    def test_quantile(self):
        img = ants.image_read(ants.get_data('r16'))
        ants.rank_intensity(img)

    def test_ilr(self):
        nsub = 20
        mu, sigma = 0, 1
        outcome = np.random.normal( mu, sigma, nsub )
        covar = np.random.normal( mu, sigma, nsub )
        mat = np.random.normal( mu, sigma, (nsub, 500 ) )
        mat2 = np.random.normal( mu, sigma, (nsub, 500 ) )
        data = {'covar':covar,'outcome':outcome}
        df = pd.DataFrame( data )
        vlist = { "mat1": mat, "mat2": mat2 }
        myform = " outcome ~ covar * mat1 "
        result = ants.ilr( df, vlist, myform)
        myform = " mat2 ~ covar + mat1 "
        result = ants.ilr( df, vlist, myform)

    def test_quantile(self):
        img = ants.image_read(ants.get_data('r16'))
        ants.quantile(img, 0.5)
        ants.quantile(img, (0.5, 0.75))

    def test_bandpass(self):
        brainSignal = np.random.randn( 400, 1000 )
        tr = 1
        filtered = ants.bandpass_filter_matrix( brainSignal, tr = tr )

    def test_compcorr(self):
        cc = ants.compcor( ants.image_read(ants.get_ants_data("ch2")) )

    def test_histogram_match(self):
        src_img = ants.image_read(ants.get_data('r16'))
        ref_img = ants.image_read(ants.get_data('r64'))
        src_ref = ants.histogram_match_image(src_img, ref_img)

        src_img = ants.image_read(ants.get_data('r16'))
        ref_img = ants.image_read(ants.get_data('r64'))
        src_ref = ants.histogram_match_image2(src_img, ref_img)

    def test_averaging(self):
        x0=[ ants.get_data('r16'), ants.get_data('r27'), ants.get_data('r62'), ants.get_data('r64') ]
        x1=[]
        for k in range(len(x0)):
            x1.append( ants.image_read( x0[k] ) )
        avg=ants.average_images(x0)
        avg1=ants.average_images(x1)
        avg2=ants.average_images(x1,mask=0)
        avg3=ants.average_images(x1,mask=1,normalize=True)

    def test_n3_2(self):
        image = ants.image_read( ants.get_ants_data('r16') )
        image_n3 = ants.n3_bias_field_correction2(image)

    def test_add_noise(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        noise_image = ants.add_noise_to_image(image, 'additivegaussian', (0.0, 1.0))
        noise_image = ants.add_noise_to_image(image, 'saltandpepper', (0.1, 0.0, 100.0))
        noise_image = ants.add_noise_to_image(image, 'shot', 1.0)
        noise_image = ants.add_noise_to_image(image, 'speckle', 1.0)

    def test_hessian_objectness(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        hessian = ants.hessian_objectness(image)

    def test_thin_plate_spline(self):
        points = np.array([[-50, -50]])
        deltas = np.array([[10, 10]])
        tps_field = ants.fit_thin_plate_spline_displacement_field(
        displacement_origins=points, displacements=deltas,
        origin=[0.0, 0.0], spacing=[1.0, 1.0], size=[100, 100],
        direction=np.array([[-1, 0], [0, -1]]))

    def test_multi_label_morph(self):
        img = ants.image_read(ants.get_data('r16'))
        labels = ants.get_mask(img,1,150) + ants.get_mask(img,151,225) * 2
        labels_dilated = ants.multi_label_morphology(labels, 'MD', 2)
        # should see original label regions preserved in dilated version
        # label N should have mean N and 0 variance
        print(ants.label_stats(labels_dilated, labels))

    def test_hausdorff_distance(self):
        r16 = ants.image_read( ants.get_ants_data('r16') )
        r64 = ants.image_read( ants.get_ants_data('r64') )
        s16 = ants.kmeans_segmentation( r16, 3 )['segmentation']
        s64 = ants.kmeans_segmentation( r64, 3 )['segmentation']
        stats = ants.hausdorff_distance(s16, s64)

    def test_channels_first(self):
        import ants
        image = ants.image_read(ants.get_ants_data('r16'))
        image2 = ants.image_read(ants.get_ants_data('r16'))
        img3 = ants.merge_channels([image,image2])
        img4 = ants.merge_channels([image,image2], channels_first=True)

        self.assertTrue(np.allclose(img3.numpy()[:,:,0], img4.numpy()[0,:,:]))
        self.assertTrue(np.allclose(img3.numpy()[:,:,1], img4.numpy()[1,:,:]))


if __name__ == "__main__":
    run_tests()
