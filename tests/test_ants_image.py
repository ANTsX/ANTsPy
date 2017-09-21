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

class Test_ANTsImage(unittest.TestCase):
    """
    Test ants_image.ANTsImage class
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_spacing(self):
        """
        test get/set spacing
        """
        img = ants.image_read(ants.get_ants_data('r16'))

        img.set_spacing((1,1))
        self.assertEqual(img.spacing, (1,1))


class Test_ants_image_Module(unittest.TestCase):
    """
    Test ants_image.py functions

    __all__ = ['copy_image_info',
               'set_origin',
               'get_origin',
               'set_direction',
               'get_direction',
               'set_spacing',
               'get_spacing',
               'image_physical_space_consistency',
               'image_type_cast',
               'get_neighborhood_in_mask',
               'get_neighborhood_at_voxel']
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    run_tests()
