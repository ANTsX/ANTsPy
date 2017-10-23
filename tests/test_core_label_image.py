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
import pandas as pd

import ants


class TestClass_LabelImage(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        square2d = np.ones((20,20))
        square2d[:10,:] += 1
        square2d[:,:10] += 2
        img2d = ants.from_numpy(square2d).astype('uint8')
        info2d = []
        for i in img2d.unique():
            info2d.append(['Value=%i'%i, 'ModTwo=%i'%int(i%2), 'ModThree=%i'%int(i%3)])
        info2d = pd.DataFrame(info2d, index=img2d.unique(), columns=['Value','ModTwo','ModThree'])

        self.img2d = img2d
        self.info2d = info2d

        square3d = np.ones((20,20,20))
        square3d[:10,...] += 1
        square3d[:,:10,:] += 2
        square3d[...,:10] += 3
        img3d = ants.from_numpy(square3d).astype('uint8')

        info3d = []
        for i in img3d.unique():
            info3d.append(['Value=%i'%i, 'ModTwo=%i'%int(i%2), 'ModThree=%i'%int(i%3)])
        info3d = pd.DataFrame(info2d, index=img2d.unique(), columns=['Value','ModTwo','ModThree'])

        self.img3d = img3d
        self.info3d = info3d

    def tearDown(self):
        pass

    def test_init(self):
        lbl_img2d = ants.LabelImage(label_image=self.img2d, label_info=self.info2d)
        lbl_img3d = ants.LabelImage(label_image=self.img2d, label_info=self.info2d)

    def test_init_with_template(self):
        arr2d = np.abs(np.random.randn(*self.img2d.shape))
        template2d = ants.from_numpy(arr2d)
        lbl_img2d = ants.LabelImage(label_image=self.img2d, label_info=self.info2d,
                                    template=template2d)

        arr3d = np.abs(np.random.randn(*self.img3d.shape))
        template3d = ants.from_numpy(arr3d)
        lbl_img3d = ants.LabelImage(label_image=self.img3d, label_info=self.info3d,
                                    template=template3d)

    def test__repr__(self):
        lbl_img2d = ants.LabelImage(label_image=self.img2d, label_info=self.info2d)
        lbl_img3d = ants.LabelImage(label_image=self.img2d, label_info=self.info2d)

        r1 = lbl_img2d.__repr__()
        r2 = lbl_img3d.__repr__()
        self.assertTrue(isinstance(r1,str))
        self.assertTrue(isinstance(r2,str))


if __name__ == '__main__':
    run_tests()
