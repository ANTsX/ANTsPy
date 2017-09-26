"""
Test ants.registration module

nptest.assert_allclose
self.assertEqual
self.assertTrue
"""

import os
import unittest
from common import run_tests
from context import ants
from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

class TestModule_affine_initializer(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        # test ANTsPy/ANTsR example
        fi = ants.image_read(ants.get_ants_data('r16'))
        mi = ants.image_read(ants.get_ants_data('r27'))
        txfile = ants.affine_initializer( fi, mi )
        tx = ants.read_transform(txfile, dimension=2)


class TestModule_apply_transforms(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        # test ANTsPy/ANTsR example
        fixed = ants.image_read( ants.get_ants_data('r16') )
        moving = ants.image_read( ants.get_ants_data('r64') )
        fixed = ants.resample_image(fixed, (64,64), 1, 0)
        moving = ants.resample_image(moving, (64,64), 1, 0)
        mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyN' )
        mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,
                                            transformlist=mytx['fwdtransforms'] )


class TestModule_create_jacobian_determinant_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read( ants.get_ants_data('r16'))
        mi = ants.image_read( ants.get_ants_data('r64'))
        fi = ants.resample_image(fi,(128,128),1,0)
        mi = ants.resample_image(mi,(128,128),1,0)
        mytx = ants.registration(fixed=fi , moving=mi, type_of_transform = ('SyN') )
        jac = ants.create_jacobian_determinant_image(fi,mytx['fwdtransforms'][0],1)


class TestModule_create_warped_grid(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read( ants.get_ants_data( 'r16' ) )
        mi = ants.image_read( ants.get_ants_data( 'r64' ) )
        mygr = ants.create_warped_grid( mi )
        
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform=('SyN') )
        mywarpedgrid = ants.create_warped_grid( mi, grid_directions=(False,True),
                            transform=mytx['fwdtransforms'], fixed_reference_image=fi )


class TestModule_fsl2antstransform(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fslmat = np.zeros((4,4))
        np.fill_diagonal(fslmat, 1)
        img = ants.image_read(ants.get_ants_data('ch2'))
        tx = ants.fsl2antstransform(fslmat, img, img)


class TestModule_interface(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        mi = ants.image_read(ants.get_ants_data('r64'))
        fi = ants.resample_image(fi, (60,60), 1, 0)
        mi = ants.resample_image(mi, (60,60), 1, 0)
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN' )


class TestModule_metrics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read( ants.get_ants_data('r16') ).clone('float')
        mi = ants.image_read( ants.get_ants_data('r64') ).clone('float')
        mival = ants.image_mutual_information(fi, mi) # -0.1796141


class TestModule_reflect_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read(ants.get_ants_data('r16'))
        axis = 2
        asym = ants.reflect_image(fi, axis, 'Affine')['warpedmovout']
        asym = asym - fi


class TestModule_reorient_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        ants.reorient_image(image, (1,0))


class TestModule_resample_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        fi = ants.image_read( ants.get_ants_data("r16"))
        finn = ants.resample_image(fi,(50,60),True,0)
        filin = ants.resample_image(fi,(1.5,1.5),False,1)


class TestModule_symmetrize_image(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_example(self):
        image = ants.image_read(ants.get_ants_data('r16'))
        simage = ants.symmetrize_image(image)


if __name__ == '__main__':
    run_tests()
