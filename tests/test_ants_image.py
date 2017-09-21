"""
Test ants_image.py

nptest.assert_allclose
self.assertEqual
self.assertTrue

"""

import os
import unittest
from common import run_tests
from context import ants

import numpy as np
import numpy.testing as nptest

import ants

class TestClass_ANTsImage(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestModule_ants_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_copy_image_info(self):
        img1 = ants.image_read(ants.get_ants_data('r16'))
        img1.set_spacing((6.9,9.6))
        img1.set_origin((3.6,3.6))

        img2 = ants.image_read(ants.get_ants_data('r64'))

        img3 = ants.copy_image_info(reference=img1, target=img2)

        self.assertEqual(img3.spacing, (6.9,6.9))
        self.assertEqual(img3.origin, (3.6,3.6))


if __name__ == '__main__':
    run_tests()
