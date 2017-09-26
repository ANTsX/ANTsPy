"""
Test ants_image.py

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


class TestClass_ANTsTransform(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        tx2d = ants.create_ants_transform(precision='float', dimension=2)
        tx3d = ants.create_ants_transform(precision='float', dimension=3)

        self.imgs = [img2d, img3d]
        self.txs = [tx2d, tx3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_get_precision(self):
        for tx in self.txs:
            precision = tx.precision
            self.assertEqual(precision, 'float')

    def test_get_dimension(self):
        tx2d = ants.create_ants_transform(precision='float', dimension=2)
        self.assertEqual(tx2d.dimension, 2)

        tx3d = ants.create_ants_transform(precision='float', dimension=3)
        self.assertEqual(tx3d.dimension, 3)

    def test_get_type(self):
        for tx in self.txs:
            ttype = tx.type
            self.assertEqual(ttype, 'AffineTransform')

    def test_get_pointer(self):
        for tx in self.txs:
            ptr = tx.pointer

    def test_get_parameters(self):
        for tx in self.txs:
            params = tx.parameters

    def test_get_fixed_parameters(self):
        for tx in self.txs:
            fixed_params = tx.fixed_parameters

    def test_invert(self):
        for tx in self.txs:
            txinvert = tx.invert()

    def test_apply_to_point(self):
        tx = ants.new_ants_transform()
        params = tx.parameters
        tx.set_parameters(params*2)
        pt2 = tx.apply_to_point((1,2,3)) # should be (2,4,6)

    def test_apply_to_image(self):
        img = ants.image_read(ants.get_ants_data("r16")).clone('float')
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((0.9,0,0,1.1,10,11))
        img2 = tx.apply_to_image(img, img)   


class TestModule_ants_transform(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        tx2d = ants.create_ants_transform(precision='float', dimension=2)
        tx3d = ants.create_ants_transform(precision='float', dimension=3)

        self.imgs = [img2d, img3d]
        self.txs = [tx2d, tx3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_get_ants_transform_parameters(self):
        for tx in self.txs:
            params = ants.get_ants_transform_parameters(tx)

    def test_set_ants_transform_parameters(self):
        for tx in self.txs:
            ants.set_ants_transform_parameters(tx, tx.parameters**2)

    def test_get_ants_transform_fixed_parameters(self):
        for tx in self.txs:
            params = ants.get_ants_transform_fixed_parameters(tx)

    def test_set_ants_transform_fixed_parameters(self):
        for tx in self.txs:
            ants.set_ants_transform_fixed_parameters(tx, tx.fixed_parameters**2)

    def test_apply_ants_transform_to_point(self):
        tx = ants.new_ants_transform()
        params = tx.parameters
        tx.set_parameters(params*2)
        pt2 = ants.apply_ants_transform_to_point(tx, (1,2,3)) # should be (2,4,6)

    def test_apply_ants_transform_to_image(self):
        img = ants.image_read(ants.get_ants_data("r16")).clone('float')
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((0.9,0,0,1.1,10,11))
        img2 = ants.apply_ants_transform_to_image(tx, img, img)

    def test_invert_ants_transform(self):
        img = ants.image_read(ants.get_ants_data("r16")).clone('float')
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((0.9,0,0,1.1,10,11))
        img_transformed = tx.apply_to_image(img, img)
        inv_tx = ants.invert_ants_transform(tx)
        img_orig = inv_tx.apply_to_image(img_transformed, img_transformed)

    def test_compose_ants_transforms(self):
        img = ants.image_read(ants.get_ants_data("r16")).clone('float')
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((0.9,0,0,1.1,10,11))
        inv_tx = tx.invert()
        single_tx = ants.compose_ants_transforms([tx, inv_tx])
        img_orig = single_tx.apply_to_image(img, img)

    def test_transform_index_to_physical_point(self):
        img = ants.make_image((10,10),np.random.randn(100))
        pt = ants.transform_index_to_physical_point(img, (2,2))

    def test_transform_physical_point_to_index(self):
        img = ants.make_image((10,10),np.random.randn(100))
        idx = ants.transform_physical_point_to_index(img, (2,2))
        img.set_spacing((2,2))
        idx2 = ants.transform_physical_point_to_index(img, (4,4))


if __name__ == '__main__':
    run_tests()
