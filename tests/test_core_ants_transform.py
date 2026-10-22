"""Check ANTsTransform mappings, parameters, and public convenience functions."""

import unittest
from common import run_tests

import numpy as np
import numpy.testing as nptest

import ants


class TestClass_ANTsTransform(unittest.TestCase):
    """
    Test ants.ANTsTransform class
    """
    def setUp(self):
        tx2d = ants.create_ants_transform(precision='float', dimension=2)
        tx3d = ants.create_ants_transform(precision='float', dimension=3)

        self.txs = [tx2d, tx3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test__repr__(self):
        for tx in self.txs:
            r = repr(tx)
            for value in ('ANTsTransform', tx.type, str(tx.dimension), tx.precision):
                self.assertIn(value, r)

    def test_get_precision(self):
        for tx in self.txs:
            precision = tx.precision
            self.assertEqual(precision, 'float')

    def test_get_dimension(self):
        tx2d = ants.create_ants_transform(precision='float', dimension=2)
        self.assertEqual(tx2d.dimension, 2)

        tx3d = ants.create_ants_transform(precision='float', dimension=3)
        self.assertEqual(tx3d.dimension, 3)

    def test_get_type(self):
        for tx in self.txs:
            ttype = tx.type
            self.assertEqual(ttype, 'AffineTransform')

    def test_get_pointer(self):
        for tx in self.txs:
            self.assertIsNotNone(tx.pointer)
            wrapped = ants.ANTsTransform(dimension=tx.dimension, pointer=tx.pointer)
            nptest.assert_allclose(wrapped.apply_to_point((1,) * tx.dimension),
                                   (1,) * tx.dimension)

    def test_get_parameters(self):
        for tx in self.txs:
            params = tx.parameters
            nptest.assert_array_equal(params, np.r_[np.eye(tx.dimension).ravel(),
                                                    np.zeros(tx.dimension)])

    def test_get_fixed_parameters(self):
        for tx in self.txs:
            fixed_params = tx.fixed_parameters
            nptest.assert_array_equal(fixed_params, np.zeros(tx.dimension))

    def test_invert(self):
        for tx in self.txs:
            matrix = np.diag(np.arange(tx.dimension) + 2.)
            translation = np.arange(tx.dimension) + 3.
            tx.set_parameters(np.r_[matrix.ravel(), translation])
            tx.set_fixed_parameters(np.arange(tx.dimension) + 0.5)
            txinvert = tx.invert()
            point = np.arange(tx.dimension) - 1.
            nptest.assert_allclose(txinvert.apply_to_point(tx.apply_to_point(point)),
                                   point, atol=1e-6)
            nptest.assert_allclose(tx.apply_to_point(txinvert.apply_to_point(point)),
                                   point, atol=1e-6)

    def test_apply(self):
        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = tx.apply(data=(1,2,3), data_type='point')
        nptest.assert_allclose(pt2, (7, 10, 13))

        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = tx.apply(data=(1,2,3), data_type='vector')
        nptest.assert_allclose(pt2, (2, 4, 6))

        img = ants.from_numpy(np.arange(30, dtype=np.float32).reshape(5, 6))
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((1, 0, 0, 1, 1, 0))
        img2 = tx.apply(data=img, reference=img, data_type='image')
        expected = np.zeros(img.shape)
        expected[:-1, :] = img.numpy()[1:, :]
        nptest.assert_array_equal(img2.numpy(), expected)
        self.assertEqual(img2.pixeltype, img.pixeltype)
        nptest.assert_allclose(img2.origin, img.origin)
        nptest.assert_allclose(img2.spacing, img.spacing)
        nptest.assert_allclose(img2.direction, img.direction)

    def test_apply_to_point(self):
        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = tx.apply_to_point((1,2,3))
        nptest.assert_allclose(pt2, (7, 10, 13))

    def test_apply_to_vector(self):
        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = tx.apply_to_vector((1,2,3))
        nptest.assert_allclose(pt2, (2, 4, 6))

    def test_apply_to_image(self):
        for ptype in self.pixeltypes:
            img = ants.from_numpy(np.arange(30, dtype=np.float32).reshape(5, 6)).clone(ptype)
            tx = ants.new_ants_transform(dimension=2)
            tx.set_parameters((1, 0, 0, 1, 1, 0))
            img2 = tx.apply_to_image(img, img)
            expected = np.zeros(img.shape)
            expected[:-1, :] = img.numpy()[1:, :]
            nptest.assert_array_equal(img2.numpy(), expected)
            self.assertEqual(img2.pixeltype, img.pixeltype)
            nptest.assert_allclose(img2.origin, img.origin)
            nptest.assert_allclose(img2.spacing, img.spacing)
            nptest.assert_allclose(img2.direction, img.direction)


class TestModule_ants_transform(unittest.TestCase):

    def setUp(self):
        tx2d = ants.create_ants_transform(precision='float', dimension=2)
        tx3d = ants.create_ants_transform(precision='float', dimension=3)

        self.txs = [tx2d, tx3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_get_ants_transform_parameters(self):
        for tx in self.txs:
            params = ants.get_ants_transform_parameters(tx)
            nptest.assert_array_equal(params, np.r_[np.eye(tx.dimension).ravel(),
                                                    np.zeros(tx.dimension)])

    def test_set_ants_transform_parameters(self):
        for tx in self.txs:
            matrix = np.diag(np.arange(tx.dimension) + 2.)
            translation = np.arange(tx.dimension) + 1.
            expected = np.r_[matrix.ravel(), translation]
            ants.set_ants_transform_parameters(tx, expected)
            nptest.assert_allclose(tx.parameters, expected)
            nptest.assert_allclose(tx.apply_to_point(np.ones(tx.dimension)),
                                   matrix @ np.ones(tx.dimension) + translation)

    def test_get_ants_transform_fixed_parameters(self):
        for tx in self.txs:
            params = ants.get_ants_transform_fixed_parameters(tx)
            nptest.assert_array_equal(params, np.zeros(tx.dimension))

    def test_set_ants_transform_fixed_parameters(self):
        for tx in self.txs:
            center = np.arange(tx.dimension) + 1.
            tx.set_parameters(np.r_[2 * np.eye(tx.dimension).ravel(),
                                    np.zeros(tx.dimension)])
            ants.set_ants_transform_fixed_parameters(tx, center)
            nptest.assert_allclose(tx.fixed_parameters, center)
            # Scaling about the new center leaves the center unchanged.
            nptest.assert_allclose(tx.apply_to_point(center), center)
            nptest.assert_allclose(tx.apply_to_point(center + 1), center + 2)

    def test_apply_ants_transform(self):
        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = ants.apply_ants_transform(tx, data=(1,2,3), data_type='point')
        nptest.assert_allclose(pt2, (7, 10, 13))

        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = ants.apply_ants_transform(tx, data=(1,2,3), data_type='vector')
        nptest.assert_allclose(pt2, (2, 4, 6))

        img = ants.from_numpy(np.arange(30, dtype=np.float32).reshape(5, 6))
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((1, 0, 0, 1, 1, 0))
        img2 = ants.apply_ants_transform(tx, data=img, reference=img, data_type='image')
        expected = np.zeros(img.shape)
        expected[:-1, :] = img.numpy()[1:, :]
        nptest.assert_array_equal(img2.numpy(), expected)
        self.assertEqual(img2.pixeltype, img.pixeltype)
        nptest.assert_allclose(img2.origin, img.origin)
        nptest.assert_allclose(img2.spacing, img.spacing)
        nptest.assert_allclose(img2.direction, img.direction)


    def test_apply_ants_transform_to_point(self):
        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = ants.apply_ants_transform_to_point(tx, (1,2,3))
        nptest.assert_allclose(pt2, (7, 10, 13))

    def test_apply_ants_transform_to_vector(self):
        tx = ants.new_ants_transform()
        tx.set_parameters(np.r_[np.eye(3).ravel() * 2, [5, 6, 7]])
        pt2 = ants.apply_ants_transform_to_vector(tx, (1,2,3))
        nptest.assert_allclose(pt2, (2, 4, 6))

    def test_apply_ants_transform_to_image(self):
        img = ants.from_numpy(np.arange(30, dtype=np.float32).reshape(5, 6))
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((1, 0, 0, 1, 1, 0))
        img2 = ants.apply_ants_transform_to_image(tx, img, img)
        expected = np.zeros(img.shape)
        expected[:-1, :] = img.numpy()[1:, :]
        nptest.assert_array_equal(img2.numpy(), expected)
        self.assertEqual(img2.pixeltype, img.pixeltype)
        nptest.assert_allclose(img2.origin, img.origin)
        nptest.assert_allclose(img2.spacing, img.spacing)
        nptest.assert_allclose(img2.direction, img.direction)

    def test_apply_ants_transform_to_image_displacement_field(self):
        values = np.arange(30, dtype=np.float32).reshape(5, 6)
        img = ants.from_numpy(values)
        displacements = np.zeros((5, 6, 2), dtype=np.float32)
        displacements[..., 0] = 1
        tx = ants.transform_from_displacement_field(
            ants.from_numpy(displacements, has_components=True))
        expected = np.zeros_like(values)
        expected[:-1, :] = values[1:, :]
        for deformed in (tx.apply_to_image(img, reference=img),
                         ants.apply_ants_transform_to_image(tx, img, img)):
            nptest.assert_array_equal(deformed.numpy(), expected)

    def test_invert_ants_transform(self):
        tx = ants.create_ants_transform(dimension=2, matrix=[[2, 0.5], [0, 3]],
                                        translation=(4, -2), center=(1, 2))
        inv_tx = ants.invert_ants_transform(tx)
        nptest.assert_allclose(inv_tx.apply_to_point((8, 6)), (2, 4), atol=1e-6)
        nptest.assert_allclose(tx.apply_to_point(inv_tx.apply_to_point((6, 4))),
                               (6, 4), atol=1e-6)

    def test_compose_ants_transforms(self):
        img = ants.from_numpy(np.arange(30, dtype=np.float32).reshape(5, 6))
        tx = ants.new_ants_transform(dimension=2)
        tx.set_parameters((1, 0, 0, 1, 1, 0))
        inv_tx = tx.invert()
        single_tx = ants.compose_ants_transforms([tx, inv_tx])
        img_orig = single_tx.apply_to_image(img, img)
        nptest.assert_allclose(img_orig.numpy(), img.numpy(), atol=1e-6)
        # Noncommuting transforms distinguish composition order.
        scale = ants.create_ants_transform(dimension=2, matrix=[[2, 0], [0, 3]])
        shift = ants.create_ants_transform(dimension=2, translation=(5, 7))
        composed = ants.compose_ants_transforms([scale, shift])
        nptest.assert_allclose(composed.apply_to_point((1, 2)), (7, 13))

        # different precisions
        tx1 = ants.new_ants_transform(dimension=2, precision='float')
        tx2 = ants.new_ants_transform(dimension=2, precision='double')
        with self.assertRaises(ValueError):
            single_tx = ants.compose_ants_transforms([tx1,tx2])

        # different dimensions
        tx1 = ants.new_ants_transform(dimension=2, precision='float')
        tx2 = ants.new_ants_transform(dimension=3, precision='float')
        with self.assertRaises(ValueError):
            single_tx = ants.compose_ants_transforms([tx1,tx2])


    def test_transform_index_to_physical_point(self):
        img = ants.from_numpy(np.zeros((10, 10), dtype=np.float32))
        img.set_origin((10, -3))
        img.set_spacing((2, 3))
        img.set_direction(np.array([[0, -1], [1, 0]]))
        pt = ants.transform_index_to_physical_point(img, (2, 2))
        nptest.assert_allclose(pt, (4, 1))

        # image not ANTsImage
        with self.assertRaises(ValueError):
            ants.transform_index_to_physical_point(2, (2,2))
        # index not tuple/list
        with self.assertRaises(ValueError):
            ants.transform_index_to_physical_point(img,img)
        # index not same size as dimension
        with self.assertRaises(ValueError):
            ants.transform_index_to_physical_point(img,[2]*(img.dimension+1))

    def test_transform_physical_point_to_index(self):
        img = ants.from_numpy(np.zeros((10, 10), dtype=np.float32))
        idx = ants.transform_physical_point_to_index(img, (2,2))
        nptest.assert_allclose(idx, (2, 2))
        img.set_spacing((2,2))
        idx2 = ants.transform_physical_point_to_index(img, (4,4))
        nptest.assert_allclose(idx2, (2, 2))
        img.set_origin((10, -3))
        img.set_spacing((2, 3))
        img.set_direction(np.array([[0, -1], [1, 0]]))
        nptest.assert_allclose(ants.transform_physical_point_to_index(img, (4, 1)), (2, 2))

        # image not ANTsImage
        with self.assertRaises(ValueError):
            ants.transform_physical_point_to_index(2, (2,2))
        # index not tuple/list
        with self.assertRaises(ValueError):
            ants.transform_physical_point_to_index(img,img)
        # index not same size as dimension
        with self.assertRaises(ValueError):
            ants.transform_physical_point_to_index(img,[2]*(img.dimension+1))


class TestDisplacementFieldFixedParameters(unittest.TestCase):

    def make_field(self, dimension):
        shape = (4, 5, 6)[:dimension]
        values = np.full(shape + (dimension,), 0.25, dtype=np.float32)
        direction = np.eye(dimension)
        direction[:2, :2] = [[0, -1], [1, 0]]
        return ants.from_numpy(values, origin=(3, -5, 7)[:dimension],
                               spacing=(1.5, 2, 2.5)[:dimension],
                               direction=direction, has_components=True)

    def test_get_fixed_parameters(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                field = self.make_field(dimension)
                tx = ants.transform_from_displacement_field(field)
                # ITK layout: size, origin, spacing, row-major direction.
                expected = np.r_[field.shape, field.origin, field.spacing,
                                 field.direction.ravel()]
                for actual in (tx.fixed_parameters,
                               ants.get_ants_transform_fixed_parameters(tx)):
                    self.assertEqual(actual.shape, (dimension * (dimension + 3),))
                    nptest.assert_allclose(actual, expected)
                point = np.asarray(field.origin) + field.direction @ np.asarray(field.spacing)
                nptest.assert_allclose(tx.apply_to_point(point), point + 0.25)
                nptest.assert_array_equal(ants.transform_to_displacement_field(tx).numpy(),
                                          field.numpy())

    def test_set_fixed_parameters(self):
        for dimension in (2, 3):
            for use_module_function in (False, True):
                for input_type in (list, tuple, np.asarray):
                    with self.subTest(dimension=dimension, module=use_module_function,
                                      input_type=input_type.__name__):
                        tx = ants.transform_from_displacement_field(self.make_field(dimension))
                        shape = (6, 4, 3)[:dimension]
                        origin = (-2, 4, 8)[:dimension]
                        spacing = (2.5, 1.25, 3)[:dimension]
                        direction = np.eye(dimension)
                        direction[:2, :2] = [[0, 1], [-1, 0]]
                        expected = np.r_[shape, origin, spacing, direction.ravel()]
                        if use_module_function:
                            ants.set_ants_transform_fixed_parameters(tx, input_type(expected))
                        else:
                            tx.set_fixed_parameters(input_type(expected))
                        nptest.assert_allclose(tx.fixed_parameters, expected)
                        field = ants.transform_to_displacement_field(tx)
                        self.assertEqual(field.shape, shape)
                        self.assertEqual(field.components, dimension)
                        nptest.assert_allclose(field.origin, origin)
                        nptest.assert_allclose(field.spacing, spacing)
                        nptest.assert_allclose(field.direction, direction)
                        # ITK allocates a new zero field, rather than preserving displacements.
                        nptest.assert_array_equal(field.numpy(), np.zeros(shape + (dimension,)))
                        point = np.asarray(origin) + direction @ np.asarray(spacing)
                        nptest.assert_allclose(tx.apply_to_point(point), point)


if __name__ == '__main__':
    run_tests()
