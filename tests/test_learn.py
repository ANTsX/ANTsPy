"""
Test ants.learn module

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

class TestModule_decomposition(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sparse_decom2_example(self):
        mat = np.random.randn(20, 100)
        mat2 = np.random.randn(20, 90)
        mydecom = ants.sparse_decom2(inmatrix = (mat,mat2), 
                                    sparseness=(0.1,0.3), nvecs=3, 
                                    its=3, perms=10)

    def test_initialize_eigenanatomy_example(self):
        mat = np.random.randn(4,100).astype('float32')
        init = ants.initialize_eigenanatomy(mat)

        img = ants.image_read(ants.get_ants_data('r16'))
        segs = ants.kmeans_segmentation(img, 3)
        init = ants.initialize_eigenanatomy(segs['segmentation'])


    def test_eig_seg_example(self):
        mylist = [ants.image_read(ants.get_ants_data('r16')),
                  ants.image_read(ants.get_ants_data('r27')),
                  ants.image_read(ants.get_ants_data('r85'))]
        myseg = ants.eig_seg(ants.get_mask(mylist[0]), mylist)

        mylist = [ants.image_read(ants.get_ants_data('r16')),
                  ants.image_read(ants.get_ants_data('r27')),
                  ants.image_read(ants.get_ants_data('r85'))]
        myseg = ants.eig_seg(ants.get_mask(mylist[0]), mylist, cthresh=2)

        mylist = [ants.image_read(ants.get_ants_data('r16')),
                  ants.image_read(ants.get_ants_data('r27')),
                  ants.image_read(ants.get_ants_data('r85'))]
        myseg = ants.eig_seg(ants.get_mask(mylist[0]), mylist, apply_segmentation_to_images=True)


if __name__ == '__main__':
    run_tests()