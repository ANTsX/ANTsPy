"""
Test that deterministic behavior is set
"""

import os
import unittest
from common import run_tests

import ants
import numpy as np

class Test_config(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_set_deterministic(self):
        """
        Test that the user can set deterministic behavior
        """

        ants.config.set_ants_deterministic(on=True, seed_value=345)
        self.assertTrue(os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] == "1")
        self.assertTrue(ants.config._deterministic)
        self.assertTrue(ants.config._random_seed == 345)

        r16 = ants.image_read(ants.get_ants_data('r16'))
        r64 = ants.image_read(ants.get_ants_data('r64'))
        
        reg1 = ants.registration(r16, r64, type_of_transform="antsRegistrationSyNRepro[a]")
        reg2 = ants.registration(r16, r64, type_of_transform="antsRegistrationSyNRepro[a]")
        self.assertTrue(np.sum(np.abs((reg1['warpedmovout'] - reg2['warpedmovout']).numpy())) == 0.0)

        reg3 = ants.registration(r16, r64, type_of_transform="antsRegistrationSyNRepro[a]")
        self.assertTrue(np.sum(np.abs((reg1['warpedmovout'] - reg3['warpedmovout']).numpy())) == 0.0)

if __name__ == '__main__':
    run_tests()
