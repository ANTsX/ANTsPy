"""
Test utils module
"""


 

import os
import unittest

from common import run_tests
from context import ants

import numpy as np
import numpy.testing as nptest


class TestModule_channels(unittest.TestCase):

    def setUp(self):
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
            arrlist = [ants.from_numpy(comparr[...,i].copy()) for i in range(len(comparr))]
            merged_img = ants.merge_channels(arrlist)
            nptest.assert_allclose(merged_img.numpy(), comparr)



if __name__ == '__main__':
    run_tests()

