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

    def test_create_ants_transform(self):
        for mtype in self.matrix_offset_types:
            for ptype in {'float', 'double'}:
                for ndim in {2, 3}:
                    ants.create_ants_transform(transform_type=mtype, precision=ptype, dimension=ndim)

        # supported types
        tx = ants.create_ants_transform(supported_types=True)

        # input params
        translation = (3,4,5)
        tx = ants.create_ants_transform( transform_type='Euler3DTransform', translation=translation )

        #translation = np.array([3,4,5])
        tx = ants.create_ants_transform( transform_type='Euler3DTransform', translation=translation )

        # invalid dimension
        with self.assertRaises(Exception):
            ants.create_ants_transform(dimension=5)

        # unsupported type
        with self.assertRaises(Exception):
            ants.create_ants_transform(transform_type='unsupported-type')

        # bad param arg
        with self.assertRaises(Exception):
            ants.create_ants_transform( transform_type='Euler3DTransform', translation=ants.image_read(ants.get_ants_data('r16')) )

        # bad precision
        with self.assertRaises(Exception):
            ants.create_ants_transform(precision='unsigned int')

    def test_read_write_transform(self):
        for tx in self.txs:
            filename = mktemp(suffix='.mat')
            ants.write_transform(tx, filename)
            tx2 = ants.read_transform(filename, precision='float')

        # file doesnt exist
        with self.assertRaises(Exception):
            ants.read_transform('blah-blah.mat')

    def test_from_displacement_components(self):
        vec_np = np.ndarray((2,2,3), dtype=np.float32)
        vec = ants.from_numpy(vec_np, origin=(0,0), spacing=(1,1), has_components=True)
        # should get ValueError here because the 2D vector field has 3 components
        with self.assertRaises(ValueError):
            ants.transform_from_displacement_field(vec)
        vec_np = np.ndarray((2,2,2,3), dtype=np.float32)
        vec = ants.from_numpy(vec_np, origin=(0,0,0), spacing=(1,1,1), has_components=True)
        # should work here because the 3D vector field has 3 components
        tx = ants.transform_from_displacement_field(vec)

    def test_from_displacement(self):
        fi = ants.image_read(ants.get_ants_data('r16') )
        mi = ants.image_read(ants.get_ants_data('r64') )
        fi = ants.resample_image(fi,(60,60),1,0)
        mi = ants.resample_image(mi,(60,60),1,0) # speed up
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
        # read transform, which calls transform_from_displacement_field
        atx = ants.read_transform( mytx['fwdtransforms'][0] )

    def test_to_displacement(self):
        fi = ants.image_read(ants.get_ants_data('r16') )
        mi = ants.image_read(ants.get_ants_data('r64') )
        fi = ants.resample_image(fi,(60,60),1,0)
        mi = ants.resample_image(mi,(60,60),1,0) # speed up
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = ('SyN') )
        vec = ants.image_read( mytx['fwdtransforms'][0] )
        atx = ants.transform_from_displacement_field( vec )
        field = ants.transform_to_displacement_field( atx, fi )

    def test_catch_error(self):
        with self.assertRaises(Exception):
            ants.write_transform(123, 'test.mat')


if __name__ == '__main__':
    run_tests()
