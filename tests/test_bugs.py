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

    def test_bspline_image_with_2d_weights(self):
        # see https://github.com/ANTsX/ANTsPy/issues/655
        import ants
        import numpy as np

        output_size = (256, 256)
        bspline_epsilon = 1e-4
        number_of_fitting_levels = 4

        image = ants.image_read(ants.get_ants_data("r16"))
        image = ants.resample_image(image, (100, 100), use_voxels=True)

        indices = np.meshgrid(list(range(image.shape[0])), 
                                list(range(image.shape[1])))
        indices_array = np.stack((indices[1].flatten(), 
                                indices[0].flatten()), axis=0)

        image_parametric_values = indices_array.transpose()

        weight_array = np.ones(image.shape)
        parametric_values = image_parametric_values
        scattered_data = np.atleast_2d(image.numpy().flatten()).transpose()
        weight_values = np.atleast_2d(weight_array.flatten()).transpose()

        min_parametric_values = np.min(parametric_values, axis=0)
        max_parametric_values = np.max(parametric_values, axis=0)

        spacing = np.zeros((2,))
        for d in range(2):
            spacing[d] = (max_parametric_values[d] - min_parametric_values[d]) / (output_size[d] - 1) + bspline_epsilon

        bspline_image = ants.fit_bspline_object_to_scattered_data(scattered_data, parametric_values, 
                                                                parametric_domain_origin=min_parametric_values - bspline_epsilon,
                                                                parametric_domain_spacing=spacing,
                                                                parametric_domain_size=output_size,
                                                                data_weights=weight_values,
                                                                number_of_fitting_levels=number_of_fitting_levels,
                                                                mesh_size=1)


    def test_scalar_rgb_missing(self):
        import ants
        img = ants.image_read(ants.get_data('r16'))
        with self.assertRaises(Exception):
            img_color = ants.scalar_to_rgb(img, cmap='jet')

    def test_bspline_zeros(self):
        import ants
        import numpy as np
        x = np.linspace(-4, 4, num=100) + np.random.uniform(-0.1, 0.1, 100)
        u = np.linspace(0, 1.0, num=len(x))
        scattered_data = np.expand_dims(u, axis=-1)
        parametric_data = np.expand_dims(u, axis=-1)
        spacing = 1/(len(x)-1) * 1.0
        bspline_curve = ants.fit_bspline_object_to_scattered_data(
                scattered_data, parametric_data,
                parametric_domain_origin=[0.0], parametric_domain_spacing=[spacing],
                parametric_domain_size=[len(x)], is_parametric_dimension_closed=None,
                number_of_fitting_levels=5, mesh_size=1)

        # Erroneously returns all zeros.
        self.assertNotEqual(bspline_curve.sum(), 0)

    def test_from_numpy_different_dtypes(self):
        all_dtypes = ('bool',
                      'int8',
                      'int16',
                      'int32',
                      'int64',
                      'uint16',
                      'uint64',
                      'float16')
        arr = np.random.randn(100,100)
        
        for dtype in all_dtypes:
            arr2 = arr.astype(dtype)
            img = ants.from_numpy(arr2)
            self.assertTrue(ants.is_image(img))
        
if __name__ == '__main__':
    run_tests()
