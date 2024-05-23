"""
Test that any previously reported bugs are fixed

nptest.assert_allclose
self.assertEqual
self.assertTrue

"""

import os
import unittest
from common import run_tests

import ants

import numpy as np
import numpy.testing as nptest


class Test_bugs(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_resample_returns_NaNs(self):
        """
        Test that resampling an image doesnt cause the resampled
        image to have NaNs - previously caused by resampling an
        image of type DOUBLE
        """
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img2dr = ants.resample_image(img2d, (2,2), 0, 0)

        self.assertTrue(np.sum(np.isnan(img2dr.numpy())) == 0)

        img3d = ants.image_read(ants.get_ants_data('mni'))
        img3dr = ants.resample_image(img3d, (2,2,2), 0, 0)

        self.assertTrue(np.sum(np.isnan(img3dr.numpy())) == 0)

    def test_compose_multi_type_transforms(self):
        image = ants.image_read(ants.get_ants_data("r16"))

        linear_transform = ants.create_ants_transform(transform_type=
            "AffineTransform", precision='float', dimension=image.dimension)

        displacement_field = ants.simulate_displacement_field(image,
            field_type="bspline", number_of_random_points=1000,
            sd_noise=10.0, enforce_stationary_boundary=True,
            number_of_fitting_levels=4, mesh_size=1,
            sd_smoothing=4.0)
        displacement_field_xfrm = ants.transform_from_displacement_field(displacement_field)

        xfrm = ants.compose_ants_transforms([linear_transform, displacement_field_xfrm])
        xfrm = ants.compose_ants_transforms([linear_transform, linear_transform])
        xfrm = ants.compose_ants_transforms([displacement_field_xfrm, linear_transform])


if __name__ == '__main__':
    run_tests()
