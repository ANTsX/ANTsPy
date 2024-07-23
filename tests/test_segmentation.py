"""
Test ants.registration module

nptest.assert_allclose
self.assertEqual
self.assertTrue


class TestModule_geo_seg(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data('simple'))
        image = ants.n3_bias_field_correction(image, 4)
        image = ants.n3_bias_field_correction(image, 2)
        bmk = ants.get_mask(image)
        segs = ants.kmeans_segmentation(image, 3, bmk)
        priors = segs['probabilityimages']
        seg = ants.geo_seg(image, bmk, priors)


"""


import os
import unittest
from common import run_tests
from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants

class TestModule_atropos(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        # test ANTsPy/ANTsR example
        img = ants.image_read(ants.get_ants_data('r16'))
        img = ants.resample_image(img, (64,64), 1, 0)
        mask = ants.get_mask(img)
        ants.atropos( a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )

    def test_multiple_inputs(self):
        img = ants.image_read( ants.get_ants_data("r16"))
        img =  ants.resample_image( img, (64,64), 1, 0 )
        mask = ants.get_mask(img)
        segs1 = ants.atropos( a = img, m = '[0.2,1x1]',
           c = '[2,0]',  i = 'kmeans[3]', x = mask )

        # Use probabilities from k-means seg as priors
        segs2 = ants.atropos( a = img, m = '[0.2,1x1]',
           c = '[2,0]',  i = segs1['probabilityimages'], x = mask )

        # multiple inputs
        feats = [img, ants.iMath(img,"Laplacian"), ants.iMath(img,"Grad") ]
        segs3 = ants.atropos( a = feats, m = '[0.2,1x1]',
           c = '[2,0]',  i = segs1['probabilityimages'], x = mask )


class TestModule_joint_label_fusion(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        ref = ants.image_read( ants.get_ants_data('r16'))
        ref = ants.resample_image(ref, (50,50),1,0)
        ref = ants.iMath(ref,'Normalize')
        mi = ants.image_read( ants.get_ants_data('r27'))
        mi2 = ants.image_read( ants.get_ants_data('r30'))
        mi3 = ants.image_read( ants.get_ants_data('r62'))
        mi4 = ants.image_read( ants.get_ants_data('r64'))
        mi5 = ants.image_read( ants.get_ants_data('r85'))
        refmask = ants.get_mask(ref)
        refmask = ants.iMath(refmask,'ME',2) # just to speed things up
        ilist = [mi,mi2,mi3,mi4,mi5]
        seglist = [None]*len(ilist)
        for i in range(len(ilist)):
            ilist[i] = ants.iMath(ilist[i],'Normalize')
            mytx = ants.registration(fixed=ref , moving=ilist[i] ,
                type_of_transform = ('Affine') )
            mywarpedimage = ants.apply_transforms(fixed=ref,moving=ilist[i],
                    transformlist=mytx['fwdtransforms'])
            ilist[i] = mywarpedimage
            seg = ants.threshold_image(ilist[i],'Otsu', 3)
            seglist[i] = ( seg ) + ants.threshold_image( seg, 1, 3 ).morphology( operation='dilate', radius=3 )

        r = 2
        pp = ants.joint_label_fusion(ref, refmask, ilist, r_search=2,
                    label_list=seglist, rad=[r]*ref.dimension )
        pp = ants.joint_label_fusion(ref,refmask,ilist, r_search=2, rad=2 )

    def test_max_lab_plus_one(self):
        ref = ants.image_read( ants.get_ants_data('r16'))
        ref = ants.resample_image(ref, (50,50),1,0)
        ref = ants.iMath(ref,'Normalize')
        mi = ants.image_read( ants.get_ants_data('r27'))
        mi2 = ants.image_read( ants.get_ants_data('r30'))
        mi3 = ants.image_read( ants.get_ants_data('r62'))
        mi4 = ants.image_read( ants.get_ants_data('r64'))
        mi5 = ants.image_read( ants.get_ants_data('r85'))
        refmask = ants.get_mask(ref)
        refmask = ants.iMath(refmask,'ME',2) # just to speed things up
        ilist = [mi,mi2,mi3,mi4,mi5]
        seglist = [None]*len(ilist)
        for i in range(len(ilist)):
            ilist[i] = ants.iMath(ilist[i],'Normalize')
            mytx = ants.registration(fixed=ref , moving=ilist[i] ,
                type_of_transform = ('Affine') )
            mywarpedimage = ants.apply_transforms(fixed=ref,moving=ilist[i],
                    transformlist=mytx['fwdtransforms'])
            ilist[i] = mywarpedimage
            seg = ants.threshold_image(ilist[i],'Otsu', 3)
            seglist[i] = ( seg ) + ants.threshold_image( seg, 1, 3 ).morphology( operation='dilate', radius=3 )

        r = 2
        pp = ants.joint_label_fusion(ref, refmask, ilist, r_search=2,
                    label_list=seglist, rad=[r]*ref.dimension, max_lab_plus_one=True )
        pp = ants.joint_label_fusion(ref,refmask,ilist, r_search=2, rad=2,
                                     max_lab_plus_one=True)



class TestModule_kelly_kapowski(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        img = ants.image_read( ants.get_ants_data('r16') ,2)
        img = ants.resample_image(img, (64,64),1,0)
        mask = ants.get_mask( img )
        segs = ants.kmeans_segmentation( img, k=3, kmask = mask)
        thick = ants.kelly_kapowski(s=segs['segmentation'], g=segs['probabilityimages'][1],
                                    w=segs['probabilityimages'][2], its=45,
                                    r=0.5, m=1)


class TestModule_kmeans(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        fi = ants.n3_bias_field_correction(fi, 2)
        seg = ants.kmeans_segmentation(fi, 3)


class TestModule_label_geometry_measures(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read( ants.get_ants_data('r16') )
        seg = ants.kmeans_segmentation( fi, 3 )['segmentation']
        geom = ants.label_geometry_measures(seg,fi)


class TestModule_prior_based_segmentation(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        seg = ants.kmeans_segmentation(fi,3)
        mask = ants.threshold_image(seg['segmentation'], 1, 1e15)
        priorseg = ants.prior_based_segmentation(fi, seg['probabilityimages'], mask, 0.25, 0.1, 3)


class TestModule_random(unittest.TestCase):
    
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_fuzzy_cmeans(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        mask = ants.get_mask(image)
        fuzzy = ants.fuzzy_spatial_cmeans_segmentation(image, mask, number_of_clusters=3)
        
    def test_functional_lung(self):
        image = ants.image_read(ants.get_data("mni")).resample_image((4,4,4))
        mask = image.get_mask()
        seg = ants.functional_lung_segmentation(image, mask, verbose=True,
                                                number_of_iterations=1,
                                                number_of_clusters=2,
                                                number_of_atropos_iterations=1)
        
    def test_anti_alias(self):
        img = ants.image_read(ants.get_data('r16'))
        mask = ants.get_mask(img)
        mask_aa = ants.anti_alias(mask)

if __name__ == '__main__':
    run_tests()
