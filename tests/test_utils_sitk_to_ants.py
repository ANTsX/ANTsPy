import os
import unittest
from tempfile import TemporaryDirectory
from common import run_tests

import numpy.testing as nptest

import ants
from ants.utils.sitk_to_ants import image_to_ants

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


class TestModule_sitk_to_ants(unittest.TestCase):
    def setUp(self):
        w, h, d = 32, 24, 16
        self.img = sitk.Image(w, h, d, sitk.sitkUInt8)
        self.img[:] = 0
        self.img[5:11, 7:12, 5:14] = 1
        self.img[7, 8, 3] = 2

    def tearDown(self):
        pass

    @unittest.skipIf(sitk is None, "SimpleITK is not installed")
    def test_image_to_ants(self):

        ants_img = image_to_ants(self.img)

        with TemporaryDirectory() as temp_dir:
            temp_fpath = os.path.join(temp_dir, "img.nrrd")
            ants.image_write(ants_img, temp_fpath)
            img = sitk.ReadImage(temp_fpath)

        nptest.assert_equal(
            sitk.GetArrayViewFromImage(self.img), sitk.GetArrayViewFromImage(img)
        )
        nptest.assert_almost_equal(self.img.GetOrigin(), img.GetOrigin())
        nptest.assert_almost_equal(self.img.GetSpacing(), img.GetSpacing())
        nptest.assert_almost_equal(self.img.GetDirection(), img.GetDirection())


if __name__ == "__main__":
    run_tests()
