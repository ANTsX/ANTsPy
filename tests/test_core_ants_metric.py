"""
Test ants_image.py

nptest.assert_allclose
self.assertEqual
self.assertTrue

    def test_initialize_and_get_value(self):
        for img, metric in self.metrics:
            img2 = img.clone() * 1.1
            metric.set_fixed_image(img)
            metric.set_moving_image(img2)
            metric.set_sampling()
            metric.initialize()
            value = metric.get_value()

    def test__call__(self):
        for img, metric in self.metrics:
            img2 = img.clone() * 1.1
            imgmask = img > img.mean()
            imgmask2 = img2 > img2.mean()
            val = metric(img,img2)
            val = metric(img,img2, fixed_mask=imgmask, moving_mask=imgmask2)
            val = metric(img,img2, sampling_percentage=0.8)


"""

import os
import unittest
from common import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants


class TestClass_ANTsImageToImageMetric(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        metric2d = ants.new_ants_metric(precision='float', dimension=2)
        metric3d = ants.new_ants_metric(precision='float', dimension=3)

        self.imgs = [img2d, img3d]
        self.metrics = [metric2d, metric3d]

    def tearDown(self):
        pass

    def test__repr__(self):
        for metric in self.metrics:
            r = metric.__repr__()

    def test_precision(self):
        for metric in self.metrics:
            precison = metric.precision

    def dimension(self):
        for metric in self.metrics:
            dimension = metric.dimension

    def metrictype(self):
        for metric in self.metrics:
            mtype  = metric.metrictype

    def is_vector(self):
        for metric in self.metrics:
            isvec = metric.is_vector

    def pointer(self):
        for metric in self.metrics:
            ptr = metric.pointer

    def test_set_fixed_image(self):
        # def set_fixed_image(self, image):
        for img, metric in zip(self.imgs,self.metrics):
            metric.set_fixed_image(img)

    def test_set_fixed_mask(self):
        # def set_fixed_image(self, image):
        for img, metric in zip(self.imgs,self.metrics):
            mask = img > img.mean()
            metric.set_fixed_mask(img)

    def test_set_moving_image(self):
        # def set_fixed_image(self, image):
        for img, metric in zip(self.imgs,self.metrics):
            metric.set_moving_image(img)

    def test_set_moving_mask(self):
        # def set_fixed_image(self, image):
        for img, metric in zip(self.imgs,self.metrics):
            mask = img > img.mean()
            metric.set_moving_mask(img)

    def test_set_sampling(self):
        for img, metric in zip(self.imgs,self.metrics):

            with self.assertRaises(Exception):
                # set sampling without moving+fixed images
                metric.set_sampling()
            
            img2 = img.clone()
            metric.set_fixed_image(img)
            metric.set_moving_image(img2)
            metric.set_sampling()

    def test_initialize(self):
        for img, metric in zip(self.imgs,self.metrics):

            with self.assertRaises(Exception):
                # initialize without moving+fixed images
                metric.initialize()
            
            img2 = img.clone()
            metric.set_fixed_image(img)
            metric.set_moving_image(img2)
            metric.set_sampling()
            metric.initialize()

    def test_get_value(self):
        for img, metric in zip(self.imgs,self.metrics):

            with self.assertRaises(Exception):
                # initialize without moving+fixed images
                metric.get_value()
            
            img2 = img.clone()
            metric.set_fixed_image(img)
            metric.set_moving_image(img2)
            metric.set_sampling()
            metric.get_value()

    def test__call__(self):
        for img, metric in zip(self.imgs,self.metrics):
            img2 = img.clone() * 1.1
            imgmask = img > img.mean()
            imgmask2 = img2 > img2.mean()
            val = metric(img,img2)
            # val = metric(img,img2, fixed_mask=imgmask, moving_mask=imgmask2)
            val = metric(img,img2, sampling_percentage=0.8)


if __name__ == '__main__':
    run_tests()
