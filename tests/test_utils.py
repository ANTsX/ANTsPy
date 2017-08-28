"""
Test utils module
"""


 

import os
import unittest

from common import run_tests
from context import ants

import numpy as np
import numpy.testing as nptest
import itk

def _parse_pixeltype_from_string(s):
    """
    example: "image_FLOAT_3D.nii.gz" -> 'float'
    """
    if 'UNSIGNEDCHAR' in s:
        return 'unsigned char'
    elif 'CHAR' in s:
        return 'char'
    elif 'UNSIGNEDSHORT' in s:
        return 'unsigned short'
    elif 'SHORT' in s:
        return 'short'
    elif 'UNSIGNEDINT' in s:
        return 'unsigned int'
    elif 'INT' in s:
        return 'int'
    elif 'FLOAT' in s:
        return 'float'
    elif 'DOUBLE' in s:
        return 'double'
    else:
        raise ValueError('cant parse pixeltype from string %s' % s)

def _parse_numpy_pixeltype_from_string(s):
    """
    example: "image_FLOAT_3D.nii.gz" -> 'float32'
    """
    if 'UNSIGNEDCHAR' in s:
        return 'uint8'
    elif 'CHAR' in s:
        return 'int8'
    elif 'UNSIGNEDSHORT' in s:
        return 'uint16'
    elif 'SHORT' in s:
        return 'int16'
    elif 'UNSIGNEDINT' in s:
        return 'uint32'
    elif 'INT' in s:
        return 'int32'
    elif 'FLOAT' in s:
        return 'float32'
    elif 'DOUBLE' in s:
        return 'float64'
    else:
        raise ValueError('cant parse pixeltype from string %s' % s)

def _parse_dim_from_string(filepath):
    """
    example: "image_FLOAT_3D.nii.gz" -> 3
    """
    if '1D' in filepath:
        return 1
    elif '2D' in filepath:
        return 2
    elif '3D' in filepath:
        return 3
    elif '4D' in filepath:
        return 4
    else:
        raise ValueError('cant parse dim from string %s' % filepath)

def _get_larger_dtype(p1, p2):
    if ( (p1=="float64") or (p2=="float64")):
        return "float64"
    elif ( (p1=="float32") or (p2=="float32")):
        return "float32"
    elif ( (p1=="int32") or (p2=="int32")):
        return "int32"
    elif ( (p1=="uint32") or (p2=="uint32")):
        return "uint32"
    elif ( (p1=="int16") or (p2=="int16")):
        return "int16"
    elif ( (p1=="uint16") or (p2=="uint16")):
        return "uint16"
    elif ( (p1=="int8") or (p2=="int8")):
        return "int8"
    elif ( (p1=="uint8") or (p2=="uint8")):
        return "uint8"




class TestUtils(unittest.TestCase):

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        img_paths = []
        for dtype in ['CHAR', 'DOUBLE', 'FLOAT', 'INT', 'SHORT', 
                      'UNSIGNEDCHAR', 'UNSIGNEDINT', 'UNSIGNEDSHORT']:
            for dim in [2,3]:
                img_paths.append('image_%s_%iD.nii.gz' % (dtype, dim))

        self.img_paths = [os.path.join(self.datadir, img_path) for img_path in img_paths]
        self.imgs = [ants.image_read(ip) for ip in self.img_paths]


    def tearDown(self):
        pass

    def test_get_mask(self): 
        low_thresh = 100
        high_thresh = 200
        cleanup = 2
        for img in self.imgs:
            mask_img = ants.get_mask(img, low_thresh=low_thresh, 
                                     high_thresh=high_thresh, cleanup=cleanup)

            # images should not be the same after bias correction
            self.assertNotEqual(img.numpy().sum(), img_n3.numpy().sum())

            # but they should have the same properties
            self.assertEqual(img.pixeltype, img_n3.pixeltype)
            self.assertEqual(img.shape, img_n3.shape)
            self.assertEqual(img.dimension, img_n3.dimension)

            self.assertEqual(img.spacing, img_n3.spacing)
            self.assertEqual(img.origin, img_n3.origin)
            nptest.assert_allclose(img.direction, img_n3.direction)

if __name__ == '__main__':
    run_tests()

