"""
Test utils module
"""


 

import os
import unittest

from common import run_tests

import numpy as np
import numpy.testing as nptest

import ants

class TestModule_bias_correction(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_n3_bias_field_correction_example(self):
        image = ants.image_read( ants.get_ants_data('r16') )
        image_n3 = ants.n3_bias_field_correction(image)

    def test_n3_bias_field_correction(self):
        for img in self.imgs:
            image_n3 = ants.n3_bias_field_correction(img)

    def test_n4_bias_field_correction_example(self):
        image = ants.image_read( ants.get_ants_data('r16') )
        image_n4 = ants.n4_bias_field_correction(image)

    def test_n4_bias_field_correction(self):
        #def n4_bias_field_correction(image, mask=None, shrink_factor=4,
        #                     convergence={'iters':[50,50,50,50], 'tol':1e-07},
        #                     spline_param=200, verbose=False, weight_mask=None):
        for img in self.imgs:
            image_n4 = ants.n4_bias_field_correction(img)
            mask = img > img.mean()
            image_n4 = ants.n4_bias_field_correction(img, mask=mask)

    def test_abp_n4_example(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        image2 = ants.abp_n4(image)

    def test_abp_n4(self):
        # def abp_n4(image, intensity_truncation=(0.025,0.975,256), mask=None, usen3=False):
        for img in self.imgs:
            img2 = ants.abp_n4(img)


class TestModule_channels(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]
        arr2d = np.random.randn(69,70,4).astype('float32')
        arr3d = np.random.randn(69,70,71,4).astype('float32')
        self.comparrs = [arr2d, arr3d]

    def tearDown(self):
        pass

    def test_merge_channels(self):
        for img in self.imgs:
            imglist = [img.clone(),img.clone()]
            merged_img = ants.merge_channels(imglist)

            self.assertEqual(merged_img.shape, img.shape)
            self.assertEqual(merged_img.components, len(imglist))
            nptest.assert_allclose(merged_img.numpy()[...,0], imglist[0].numpy())

            imglist = [img.clone(),img.clone(),img.clone(),img.clone()]
            merged_img = ants.merge_channels(imglist)

            self.assertEqual(merged_img.shape, img.shape)
            self.assertEqual(merged_img.components, len(imglist))
            nptest.assert_allclose(merged_img.numpy()[...,0], imglist[-1].numpy())

        for comparr in self.comparrs:
            imglist = [ants.from_numpy(comparr[...,i].copy()) for i in range(comparr.shape[-1])]
            merged_img = ants.merge_channels(imglist)
            for i in range(comparr.shape[-1]):
                nptest.assert_allclose(merged_img.numpy()[...,i], comparr[...,i])


class TestModule_crop_image(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_crop_image_example(self):
        fi = ants.image_read( ants.get_ants_data('r16') )
        cropped = ants.crop_image(fi)
        cropped = ants.crop_image(fi, fi, 100 )

    def test_crop_indices_example(self):
        fi = ants.image_read( ants.get_ants_data("r16"))
        cropped = ants.crop_indices( fi, (10,10), (100,100) )
        cropped = ants.smooth_image( cropped, 5 )
        decropped = ants.decrop_image( cropped, fi )

    def test_decrop_image_example(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        mask = ants.get_mask(fi)
        cropped = ants.crop_image(fi, mask, 1)
        cropped = ants.smooth_image(cropped, 1)
        decropped = ants.decrop_image(cropped, fi)


class TestModule_denoise_image(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_denoise_image_example(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        # add fairly large salt and pepper noise
        imagenoise = image + np.random.randn(*image.shape).astype('float32')*5
        imagedenoise = ants.denoise_image(imagenoise, ants.get_mask(image))


class TestModule_get_ants_data(unittest.TestCase):

    def test_get_ants_data(self):
        for imgpath in ants.get_ants_data_files():
            img = ants.image_read(imgpath)

    def test_get_ants_data_files(self):
        imgpaths = ants.get_ants_data_files()
        self.assertTrue(isinstance(imgpaths,list))
        self.assertTrue(len(imgpaths) > 0)


class TestModule_get_centroids(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_get_centroids_example(self):
        image = ants.image_read( ants.get_ants_data( "r16" ) )
        image = ants.threshold_image( image, 90, 120 )
        image = ants.label_clusters( image, 10 )
        cents = ants.get_centroids( image  )


class TestModule_get_mask(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_get_mask_example(self):
        image = ants.image_read( ants.get_ants_data('r16') )
        mask = ants.get_mask(image)


class TestModule_get_neighborhood(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_get_neighborhood_in_mask_example(self):
        r16 = ants.image_read(ants.get_ants_data('r16'))
        mask = ants.get_mask(r16)
        mat = ants.get_neighborhood_in_mask(r16, mask, radius=(2,2))

    def test_get_neighborhood_at_voxel_example(self):
        img = ants.image_read(ants.get_ants_data('r16'))
        center = (2,2)
        radius = (3,3)
        retval = ants.get_neighborhood_at_voxel(img, center, radius)


class TestModule_image_similarity(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_image_similarity_example(self):
        x = ants.image_read(ants.get_ants_data('r16'))
        y = ants.image_read(ants.get_ants_data('r30'))
        metric = ants.image_similarity(x,y,metric_type='MeanSquares')

class TestModule_image_to_cluster_images(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_image_to_cluster_images_example(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        image = ants.threshold_image(image, 1, 1e15)
        image_cluster_list = ants.image_to_cluster_images(image)


class TestModule_iMath(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni')).resample_image((2,2,2))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_iMath_example(self):
        img = ants.image_read(ants.get_ants_data('r16'))
        img2 = ants.iMath(img, 'Canny', 1, 5, 12)


class TestModule_impute(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_impute_example(self):
        data = np.random.randn(4,10)
        data[2,3] = np.nan
        data[3,5] = np.nan
        data_imputed = ants.impute(data, 'mean')


class TestModule_invariant_image_similarity(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_invariate_image_similarity_example(self):
        img1 = ants.image_read(ants.get_ants_data('r16'))
        img2 = ants.image_read(ants.get_ants_data('r64'))
        metric1 = ants.invariant_image_similarity(img1,img2, do_reflection=False)

        img1 = ants.image_read(ants.get_ants_data('r16'))
        img2 = ants.image_read(ants.get_ants_data('r64'))
        metric2 = ants.invariant_image_similarity(img1,img2, do_reflection=True)

    def test_convolve_image_example(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        convimg = ants.make_image( (3,3), (1,0,1,0,-4,0,1,0,1) )
        convout = ants.convolve_image( fi, convimg )
        convimg2 = ants.make_image( (3,3), (0,1,0,1,0,-1,0,-1,0) )
        convout2 = ants.convolve_image( fi, convimg2 )


class TestModule_label_clusters(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_label_clusters_example(self):
        image = ants.image_read( ants.get_ants_data('r16') )
        timageFully = ants.label_clusters( image, 10, 128, 150, True )
        timageFace = ants.label_clusters( image, 10, 128, 150, False )


class TestModule_label_image_centroids(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_label_clusters_example(self):
        image = ants.from_numpy(np.asarray([[[0,2],[1,3]],[[4,6],[5,7]]]).astype('float32'))
        labels = ants.label_image_centroids(image)


class TestModule_label_stats(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_label_stats_example(self):
        image = ants.image_read( ants.get_ants_data('r16') , 2 )
        image = ants.resample_image( image, (64,64), 1, 0 )
        mask = ants.get_mask(image)
        segs1 = ants.kmeans_segmentation( image, 3 )
        stats = ants.label_stats(image, segs1['segmentation'])



class TestModule_labels_to_matrix(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_labels_to_matrix_example(self):
        fi = ants.image_read(ants.get_ants_data('r16')).resample_image((60,60),1,0)
        mask = ants.get_mask(fi)
        labs = ants.kmeans_segmentation(fi,3)['segmentation']
        labmat = ants.labels_to_matrix(labs, mask)



class TestModule_mask_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mask_image_example(self):
        myimage = ants.image_read(ants.get_ants_data('r16'))
        mask = ants.get_mask(myimage)
        myimage_mask = ants.mask_image(myimage, mask, 3)
        seg = ants.kmeans_segmentation(myimage, 3)
        myimage_mask = ants.mask_image(myimage, seg['segmentation'], (1,3))


class TestModule_mni2tal(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mni2tal_example(self):
        ants.mni2tal( (10,12,14) )


class TestModule_morphology(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_morphology_example(self):
        fi = ants.image_read( ants.get_ants_data('r16') , 2 )
        mask = ants.get_mask( fi )
        dilated_ball = ants.morphology( mask, operation='dilate', radius=3, mtype='binary', shape='ball')
        eroded_box = ants.morphology( mask, operation='erode', radius=3, mtype='binary', shape='box')
        opened_annulus = ants.morphology( mask, operation='open', radius=5, mtype='binary', shape='annulus', thickness=2)



class TestModule_smooth_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_smooth_image_example(self):
        image = ants.image_read( ants.get_ants_data('r16'))
        simage = ants.smooth_image(image, (1.2,1.5))


class TestModule_threshold_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_threshold_image_example(self):
        image = ants.image_read( ants.get_ants_data('r16') )
        timage = ants.threshold_image(image, 0.5, 1e15)


class TestModule_weingarten_image_curvature(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_weingarten_image_curvature_example(self):
        image = ants.image_read(ants.get_ants_data('mni')).resample_image((3,3,3))
        imagecurv = ants.weingarten_image_curvature(image)

        image2 = ants.image_read(ants.get_ants_data('r16')).resample_image((2,2))
        imagecurv2 = ants.weingarten_image_curvature(image2)


if __name__ == '__main__':
    run_tests()

