"""Check transform construction, serialization, and displacement field conversion."""

import os
import unittest
from common import run_tests

from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as nptest

import ants


class TestModule_ants_transform_io(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        tx2d = ants.create_ants_transform(precision='float', dimension=2)
        tx3d = ants.create_ants_transform(precision='float', dimension=3)

        self.imgs = [img2d, img3d]
        self.txs = [tx2d, tx3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

        self.matrix_offset_types = ['AffineTransform',
                                    'CenteredAffineTransform',
                                    'Euler2DTransform',
                                    'Euler3DTransform',
                                    'Rigid2DTransform',
                                    'QuaternionRigidTransform',
                                    'Similarity2DTransform',
                                    'CenteredSimilarity2DTransform',
                                    'Similarity3DTransform',
                                    'CenteredRigid2DTransform',
                                    'CenteredEuler3DTransform',
                                    'Rigid3DTransform']

    def tearDown(self):
        pass

    def test_new_ants_transform(self):
        tx = ants.new_ants_transform()
        params = tx.parameters*2

        # initialize w/ params
        tx = ants.new_ants_transform(parameters=params)
        nptest.assert_array_equal(tx.parameters, params)
        nptest.assert_allclose(tx.apply_to_point((1, 2, 3)), (2, 4, 6))

    def test_create_ants_transform(self):
        for mtype in self.matrix_offset_types:
            for ptype in {'float', 'double'}:
                for ndim in {2, 3}:
                    with self.subTest(transform_type=mtype, precision=ptype, dimension=ndim):
                        tx = ants.create_ants_transform(transform_type=mtype, precision=ptype, dimension=ndim)
                        self.assertEqual(tx.type, mtype)
                        self.assertEqual(tx.precision, ptype)
                        expected_dimension = ndim
                        if '2D' in mtype:
                            expected_dimension = 2
                        elif '3D' in mtype or mtype == 'QuaternionRigidTransform':
                            expected_dimension = 3
                        self.assertEqual(tx.dimension, expected_dimension)
                        nptest.assert_allclose(tx.apply_to_point((1,) * tx.dimension),
                                               (1,) * tx.dimension)

        # supported types
        tx = ants.create_ants_transform(supported_types=True)
        self.assertEqual(tx, set(self.matrix_offset_types) | {'DisplacementFieldTransform'})

        # input params
        translation = (3,4,5)
        tx = ants.create_ants_transform( transform_type='Euler3DTransform', translation=translation )
        nptest.assert_allclose(tx.apply_to_point((1, 2, 3)), (4, 6, 8))

        translation = np.array([3, 4, 5])
        tx = ants.create_ants_transform( transform_type='Euler3DTransform', translation=translation )
        nptest.assert_allclose(tx.apply_to_point((1, 2, 3)), (4, 6, 8))

        # invalid dimension
        with self.assertRaises(ValueError):
            ants.create_ants_transform(dimension=5)

        # unsupported type
        with self.assertRaises(ValueError):
            ants.create_ants_transform(transform_type='unsupported-type')

        # bad param arg
        with self.assertRaises(ValueError):
            ants.create_ants_transform( transform_type='Euler3DTransform', translation=ants.image_read(ants.get_ants_data('r16')) )

        # bad precision
        with self.assertRaises(ValueError):
            ants.create_ants_transform(precision='unsigned int')

    def test_read_write_transform(self):
        with TemporaryDirectory() as directory:
            for tx in self.txs:
                tx.set_parameters(np.r_[2 * np.eye(tx.dimension).ravel(),
                                        np.arange(tx.dimension) + 1])
                tx.set_fixed_parameters(np.arange(tx.dimension) + 0.5)
                filename = os.path.join(directory, 'transform.mat')
                ants.write_transform(tx, filename)
                tx2 = ants.read_transform(filename, precision='float')
                self.assertEqual(tx2.type, tx.type)
                self.assertEqual(tx2.dimension, tx.dimension)
                nptest.assert_allclose(tx2.parameters, tx.parameters)
                nptest.assert_allclose(tx2.fixed_parameters, tx.fixed_parameters)
                nptest.assert_allclose(tx2.apply_to_point(np.ones(tx.dimension)),
                                       np.full(tx.dimension, 2.5))

        # file doesnt exist
        with self.assertRaises(ValueError):
            ants.read_transform('blah-blah.mat')

    def test_from_displacement_components(self):
        vec_np = np.zeros((2,2,3), dtype=np.float32)
        vec = ants.from_numpy(vec_np, origin=(0,0), spacing=(1,1), has_components=True)
        # should get ValueError here because the 2D vector field has 3 components
        with self.assertRaises(ValueError):
            ants.transform_from_displacement_field(vec)
        vec_np = np.full((2,2,2,3), 0.25, dtype=np.float32)
        vec = ants.from_numpy(vec_np, origin=(0,0,0), spacing=(1,1,1), has_components=True)
        # should work here because the 3D vector field has 3 components
        tx = ants.transform_from_displacement_field(vec)
        self.assertEqual(tx.type, 'DisplacementFieldTransform')
        self.assertEqual(tx.dimension, 3)
        nptest.assert_allclose(tx.apply_to_point((0.5, 0.5, 0.5)), (0.75, 0.75, 0.75))

    def test_from_displacement(self):
        fi = ants.image_read(ants.get_ants_data('r16') )
        mi = ants.image_read(ants.get_ants_data('r64') )
        fi = ants.resample_image(fi,(60,60),1,0)
        mi = ants.resample_image(mi,(60,60),1,0) # speed up
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
        # read transform, which calls transform_from_displacement_field
        atx = ants.read_transform( mytx['fwdtransforms'][0] )
        self.assertEqual(atx.type, 'DisplacementFieldTransform')
        expected = ants.image_read(mytx['fwdtransforms'][0])
        nptest.assert_allclose(ants.transform_to_displacement_field(atx).numpy(), expected.numpy())

    def test_to_displacement(self):
        fi = ants.image_read(ants.get_ants_data('r16') )
        mi = ants.image_read(ants.get_ants_data('r64') )
        fi = ants.resample_image(fi,(60,60),1,0)
        mi = ants.resample_image(mi,(60,60),1,0) # speed up
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
        vec = ants.image_read( mytx['fwdtransforms'][0] )
        atx = ants.transform_from_displacement_field( vec )
        with self.assertWarnsRegex(DeprecationWarning, "ref.*deprecated"):
            field = ants.transform_to_displacement_field(atx, fi)
        nptest.assert_allclose(field.numpy(), vec.numpy())
        nptest.assert_allclose(field.origin, fi.origin)
        nptest.assert_allclose(field.spacing, fi.spacing)
        nptest.assert_allclose(field.direction, fi.direction)
        self.assertEqual(field.shape, fi.shape)

    def test_catch_error(self):
        with self.assertRaisesRegex(Exception, 'Only ANTsTransform instances'):
            ants.write_transform(123, 'test.mat')


class TestTransformFileRoundTrips(unittest.TestCase):

    def make_field(self, dimension):
        shape = (6,) * dimension
        indices = np.indices(shape, dtype=np.float32)
        values = np.moveaxis(indices, 0, -1) * 0.02 + 0.1
        direction = np.eye(dimension)
        direction[:2, :2] = [[0, -1], [1, 0]]
        return ants.from_numpy(values, origin=(3., -5., 7.)[:dimension],
                               spacing=(1.5, 2., 2.5)[:dimension], direction=direction,
                               has_components=True)

    def points_in_field(self, field):
        return [tuple(np.asarray(field.origin) + field.direction @
                      (np.asarray(index) * field.spacing))
                for index in [(1.,) * field.dimension, (2.5,) * field.dimension,
                              (4.,) * field.dimension]]

    def assert_mapping_equal(self, expected, actual, points):
        self.assertEqual(actual.dimension, expected.dimension)
        for point in points:
            nptest.assert_allclose(actual.apply_to_point(point),
                                   expected.apply_to_point(point), rtol=1e-6, atol=1e-5)

    def test_affine_formats(self):
        with TemporaryDirectory() as directory:
            for dimension in (2, 3):
                for precision in ('float', 'double'):
                    tx = ants.create_ants_transform(
                        dimension=dimension, precision=precision,
                        matrix=np.diag(np.arange(dimension) * 0.1 + 0.9),
                        translation=np.arange(dimension) + 0.5,
                        center=np.arange(dimension) + 2.)
                    for suffix in ('.txt', '.mat', '.h5', '.hdf5'):
                        with self.subTest(dimension=dimension, precision=precision, suffix=suffix):
                            path = os.path.join(directory, 'affine' + suffix)
                            ants.write_transform(tx, path)
                            restored = ants.read_transform(path, precision=precision)
                            self.assertEqual(restored.type, 'AffineTransform')
                            nptest.assert_allclose(restored.parameters, tx.parameters)
                            nptest.assert_allclose(restored.fixed_parameters, tx.fixed_parameters)
                            self.assert_mapping_equal(tx, restored, [(0.,) * dimension,
                                                                    (2.,) * dimension])

    def test_displacement_formats(self):
        with TemporaryDirectory() as directory:
            for dimension in (2, 3):
                field = self.make_field(dimension)
                tx = ants.transform_from_displacement_field(field)
                points = self.points_in_field(field)
                for suffix in ('.nii', '.nii.gz', '.h5', '.hdf5'):
                    with self.subTest(dimension=dimension, suffix=suffix):
                        path = os.path.join(directory, 'warp' + suffix)
                        ants.write_transform(tx, path)
                        restored = ants.read_transform(path)
                        self.assert_mapping_equal(tx, restored, points)
                        if suffix in ('.nii', '.nii.gz'):
                            restored_field = ants.image_read(path)
                            nptest.assert_array_equal(restored_field.numpy(), field.numpy())
                            nptest.assert_allclose(restored_field.origin, field.origin)
                            nptest.assert_allclose(restored_field.spacing, field.spacing)
                            nptest.assert_allclose(restored_field.direction, field.direction)
                            self.assertEqual(restored_field.components, dimension)
                            self.assert_mapping_equal(
                                tx, ants.transform_from_displacement_field(restored_field), points)
                        else:
                            self.assertEqual(restored.type, 'CompositeTransform')

    def test_composite_hdf5(self):
        with TemporaryDirectory() as directory:
            for dimension in (2, 3):
                field = self.make_field(dimension)
                warp = ants.transform_from_displacement_field(field)
                affine = ants.create_ants_transform(dimension=dimension,
                                                    translation=(0.25,) * dimension)
                composite = ants.compose_ants_transforms([affine, warp])
                for suffix in ('.h5', '.hdf5'):
                    with self.subTest(dimension=dimension, suffix=suffix):
                        path = os.path.join(directory, 'composite' + suffix)
                        ants.write_transform(composite, path)
                        restored = ants.read_transform(path)
                        self.assertEqual(restored.type, 'CompositeTransform')
                        self.assert_mapping_equal(composite, restored, self.points_in_field(field))

    def test_nifti_rejects_non_displacement_transforms(self):
        affine = ants.new_ants_transform(dimension=2)
        composite = ants.compose_ants_transforms([affine])
        with TemporaryDirectory() as directory:
            for tx in (affine, composite):
                for suffix in ('.nii', '.nii.gz'):
                    path = os.path.join(directory, 'invalid' + suffix)
                    with self.assertRaisesRegex(ValueError, 'DisplacementFieldTransform'):
                        ants.write_transform(tx, path)
                    self.assertFalse(os.path.exists(path))


class TestDisplacementFieldExtraction(unittest.TestCase):

    def test_stored_geometry_and_explicit_reference(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                shape = (4, 5) if dimension == 2 else (4, 5, 3)
                values = np.arange(np.prod(shape) * dimension, dtype=np.float32)
                values = values.reshape(shape + (dimension,)) / 10
                direction = np.eye(dimension)
                direction[:2, :2] = [[0, -1], [1, 0]]
                field = ants.from_numpy(
                    values, origin=tuple(range(2, dimension + 2)),
                    spacing=tuple(1.5 + i for i in range(dimension)),
                    direction=direction, has_components=True,
                )
                tx = ants.transform_from_displacement_field(field)
                ref = ants.from_numpy(
                    np.zeros(shape, dtype=np.float32), origin=field.origin,
                    spacing=field.spacing, direction=field.direction,
                )
                with self.assertWarnsRegex(DeprecationWarning, "ref.*deprecated"):
                    field_with_reference = ants.transform_to_displacement_field(tx, ref)
                for result in (ants.transform_to_displacement_field(tx),
                               ants.transform_to_displacement_field(tx, None),
                               field_with_reference):
                    nptest.assert_array_equal(result.numpy(), values)
                    self.assertEqual(result.shape, field.shape)
                    self.assertEqual(result.components, dimension)
                    self.assertEqual(result.pixeltype, "float")
                    nptest.assert_allclose(result.origin, field.origin)
                    nptest.assert_allclose(result.spacing, field.spacing)
                    nptest.assert_allclose(result.direction, field.direction)

                # Explicit references still supply geometry without resampling.
                ref.set_origin(tuple(20 + i for i in range(dimension)))
                with self.assertWarnsRegex(DeprecationWarning, "ref.*deprecated"):
                    result = ants.transform_to_displacement_field(tx, ref)
                nptest.assert_allclose(result.origin, ref.origin)
                nptest.assert_array_equal(result.numpy(), values)

    def test_rejects_affine_without_reference(self):
        with self.assertRaisesRegex(ValueError, "DisplacementFieldTransform"):
            ants.transform_to_displacement_field(ants.new_ants_transform(dimension=2))


if __name__ == '__main__':
    run_tests()
