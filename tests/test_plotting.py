import os
import unittest
from common import run_tests
from tempfile import mktemp

# Prevent displaying figures
import matplotlib as mpl
backend_ =  mpl.get_backend() 
mpl.use("Agg")  

import numpy as np
import numpy.testing as nptest

import ants


class TestModule_plot(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_exceptions(self):
        # non image
        with self.assertRaises(Exception):
            ants.plot(123)
            
        # image with components
        image = ants.image_read(ants.get_ants_data('r16'))
        image = ants.merge_channels([image,image])
        with self.assertRaises(Exception):
            ants.plot(image)
            
        # 4d image

    def test_plot_2d(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        
        ants.plot(image)
        
        ants.plot()
    
    def test_plot_3d(self):
        image = ants.image_read(ants.get_ants_data('mni'))

    
if __name__ == '__main__':
    run_tests()
