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





class Test_ANTsImage(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    run_tests()
