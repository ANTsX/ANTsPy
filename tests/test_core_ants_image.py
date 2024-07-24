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


class TestClass_ANTsImage(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_get_spacing(self):
        #self.setUp()
        for img in self.imgs:
            spacing = img.spacing
            self.assertTrue(isinstance(spacing, tuple))
            self.assertEqual(len(img.spacing), img.dimension)

    def test_set_spacing(self):
        #self.setUp()
        for img in self.imgs:
            # set spacing from list
            new_spacing_list = [6.9]*img.dimension
            img.set_spacing(new_spacing_list)
            self.assertEqual(img.spacing, tuple(new_spacing_list))

            # set spacing from tuple
            new_spacing_tuple = tuple(new_spacing_list)
            img.set_spacing(new_spacing_tuple)
            self.assertEqual(img.spacing, new_spacing_tuple)

            # test exceptions
            with self.assertRaises(Exception):
                new_spacing = img.clone()
                img.set_spacing(new_spacing)
            with self.assertRaises(Exception):
                new_spacing = [6.9]*(img.dimension+1)
                img.set_spacing(new_spacing)

    def test_get_origin(self):
        #self.setUp()
        for img in self.imgs:
            origin = img.origin
            self.assertTrue(isinstance(origin, tuple))
            self.assertEqual(len(img.origin), img.dimension)

    def test_set_origin(self):
        for img in self.imgs:
            # set spacing from list
            new_origin_list = [6.9]*img.dimension
            img.set_origin(new_origin_list)
            self.assertEqual(img.origin, tuple(new_origin_list))

            # set spacing from tuple
            new_origin_tuple = tuple(new_origin_list)
            img.set_origin(new_origin_tuple)
            self.assertEqual(img.origin, new_origin_tuple)

            # test exceptions
            with self.assertRaises(Exception):
                new_origin = img.clone()
                img.set_origin(new_origin)
            with self.assertRaises(Exception):
                new_origin = [6.9]*(img.dimension+1)
                img.set_origin(new_origin)

    def test_get_direction(self):
        #self.setUp()
        for img in self.imgs:
            direction = img.direction
            self.assertTrue(isinstance(direction,np.ndarray))
            self.assertTrue(img.direction.shape, (img.dimension,img.dimension))

    def test_set_direction(self):
        #self.setUp()
        for img in self.imgs:
            new_direction = np.eye(img.dimension)*3
            img.set_direction(new_direction)
            nptest.assert_allclose(img.direction, new_direction)

            # set from list
            img.set_direction(new_direction.tolist())

            # test exceptions
            with self.assertRaises(Exception):
                new_direction = img.clone()
                img.set_direction(new_direction)
            with self.assertRaises(Exception):
                new_direction = np.eye(img.dimension+2)*3
                img.set_direction(new_direction)

    def test_view(self):
        #self.setUp()
        for img in self.imgs:
            myview = img.view()
            self.assertTrue(isinstance(myview, np.ndarray))
            self.assertEqual(myview.shape, img.shape)

            # test that changes to view are direct changes to the image
            myview *= 3.
            nptest.assert_allclose(myview, img.numpy())

    def test_numpy(self):
        #self.setUp()
        for img in self.imgs:
            mynumpy = img.numpy()
            self.assertTrue(isinstance(mynumpy, np.ndarray))
            self.assertEqual(mynumpy.shape, img.shape)

            # test that changes to view are NOT direct changes to the image
            mynumpy *= 3.
            nptest.assert_allclose(mynumpy, img.numpy()*3.)

    def test_clone(self):
        #self.setUp()
        for img in self.imgs:
            orig_ptype = img.pixeltype
            for ptype in self.pixeltypes:
                imgclone = img.clone(ptype)

                self.assertEqual(imgclone.pixeltype, ptype)
                self.assertEqual(img.pixeltype, orig_ptype)
                # test physical space consistency
                self.assertTrue(ants.image_physical_space_consistency(img, imgclone))
                if ptype == orig_ptype:
                    # test that they dont share image pointer
                    view1 = img.view()
                    view1 *= 6.9
                    nptest.assert_allclose(view1, imgclone.numpy()*6.9)

    def test_new_image_like(self):
        #self.setUp()
        for img in self.imgs:
            myarray = img.numpy()
            myarray *= 6.9
            imgnew = img.new_image_like(myarray)
            # test physical space consistency
            self.assertTrue(ants.image_physical_space_consistency(img, imgnew))
            # test data
            nptest.assert_allclose(myarray, imgnew.numpy())
            nptest.assert_allclose(myarray, img.numpy()*6.9)

            # test exceptions
            with self.assertRaises(Exception):
                # not ndarray
                new_data = img.clone()
                img.new_image_like(new_data)
            with self.assertRaises(Exception):
                # wrong shape
                new_data = np.random.randn(69,12,21).astype('float32')
                img.new_image_like(new_data)

        with self.assertRaises(Exception):
            # wrong shape with components
            vecimg = ants.from_numpy(np.random.randn(69,12,3).astype('float32'), has_components=True)
            new_data = np.random.randn(69,12,4).astype('float32')
            vecimg.new_image_like(new_data)

    def test_from_numpy_like(self):
        img = ants.image_read(ants.get_data('mni'))

        arr = img.numpy()
        arr *= 2
        img2 = ants.from_numpy_like(arr, img)
        self.assertTrue(ants.image_physical_space_consistency(img, img2))
        self.assertEqual(img2.mean() / img.mean(), 2)

    def test_to_file(self):
        #self.setUp()
        for img in self.imgs:
            filename = mktemp(suffix='.nii.gz')
            img.to_file(filename)
            # test that file now exists
            self.assertTrue(os.path.exists(filename))
            img2 = ants.image_read(filename)
            # test physical space and data
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img.numpy(), img2.numpy())

            try:
                os.remove(filename)
            except:
                pass

    def test_apply(self):
        #self.setUp()
        for img in self.imgs:
            img2 = img.apply(lambda x: x*6.9)
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()*6.9)

    def test_numpy_ops(self):
        for img in self.imgs:
            median = img.median()
            std = img.std()
            argmin = img.argmin()
            argmax = img.argmax()
            flat = img.flatten()
            nonzero = img.nonzero()
            unique = img.unique()

    def test__add__(self):
        #self.setUp()
        for img in self.imgs:
            # op on constant
            img2 = img + 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()+6.9)

            # op on another image
            img2 = img + img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()+img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img + img2

    def test__radd__(self):
        #self.setUp()
        for img in self.imgs:
            # op on constant
            img2 = img.__radd__(6.9)
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy() + 6.9)

            # op on another image
            img2 = img + img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()+img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img + img2

    def test__sub__(self):
        #self.setUp()
        for img in self.imgs:
            # op on constant
            img2 = img - 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()-6.9)

            # op on another image
            img2 = img - img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()-img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img - img2

    def test__rsub__(self):
        #self.setUp()
        for img in self.imgs:
            # op on constant
            img2 = img.__rsub__(6.9)
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), 6.9 - img.numpy())

            # op on another image
            img2 = img - img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()-img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img - img2

    def test__mul__(self):
        #self.setUp()
        for img in self.imgs:
            # op on constant
            img2 = img * 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()*6.9)

            # op on another image
            img2 = img * img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()*img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img * img2

    def test__rmul__(self):
        #self.setUp()
        for img in self.imgs:
            # op on constant
            img2 = img.__rmul__(6.9)
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), 6.9*img.numpy())

            # op on another image
            img2 = img * img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()*img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img * img2

    def test__div__(self):
        #self.setUp()
        for img in self.imgs:
            img = img + 10
            # op on constant
            img2 = img / 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()/6.9)

            # op on another image
            img2 = img / img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()/img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img / img2

    def test__pow__(self):
        #self.setUp()
        for img in self.imgs:
            img = img + 10
            # op on constant
            img2 = img ** 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()**6.9)

            # op on another image
            img2 = img ** img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()**img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img ** img2

    def test__gt__(self):
        #self.setUp()
        for img in self.imgs:
            img2 = img > 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), (img.numpy()>6.9).astype('int'))

            # op on another image
            img2 = img > img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()>img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img > img2

    def test__ge__(self):
        #self.setUp()
        for img in self.imgs:
            img2 = img >= 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), (img.numpy()>=6.9).astype('int'))

            # op on another image
            img2 = img >= img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()>=img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img >= img2

    def test__lt__(self):
        #self.setUp()
        for img in self.imgs:
            img2 = img < 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), (img.numpy()<6.9).astype('int'))

            # op on another image
            img2 = img < img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()<img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img < img2

    def test__le__(self):
        #self.setUp()
        for img in self.imgs:
            img2 = img <= 6.9
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), (img.numpy()<=6.9).astype('int'))

            # op on another image
            img2 = img <= img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()<=img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img <= img2

    def test__eq__(self):
        #self.setUp()
        for img in self.imgs:
            img2 = (img == 6.9)
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), (img.numpy()==6.9).astype('int'))

            # op on another image
            img2 = img == img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()==img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img == img2

    def test__ne__(self):
        #self.setUp()
        for img in self.imgs:
            img2 = (img != 6.9)
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), (img.numpy()!=6.9).astype('int'))

            # op on another image
            img2 = img != img.clone()
            self.assertTrue(ants.image_physical_space_consistency(img, img2))
            nptest.assert_allclose(img2.numpy(), img.numpy()!=img.numpy())

            with self.assertRaises(Exception):
                # different physical space
                img2 = img.clone()
                img2.set_spacing([2.31]*img.dimension)
                img3 = img != img2

    def test__getitem__(self):
        for img in self.imgs:
            if img.dimension == 2:
                img2 = img[6:9,6:9]
                nptest.assert_allclose(img2.numpy(), img.numpy()[6:9,6:9])
            elif img.dimension == 3:
                img2 = img[6:9,6:9,6:9]
                nptest.assert_allclose(img2.numpy(), img.numpy()[6:9,6:9,6:9])

            # get from another image
            img2 = img.clone()
            xx = img[img2]
            with self.assertRaises(Exception):
                # different physical space
                img2.set_direction(img.direction*2)
                xx = img[img2]

    def test__setitem__(self):
        for img in self.imgs:
            if img.dimension == 2:
                img[6:9,6:9] = 6.9
                self.assertTrue(img.numpy()[6:9,6:9].mean(), 6.9)
            elif img.dimension == 3:
                img[6:9,6:9,6:9] = 6.9
                self.assertTrue(img.numpy()[6:9,6:9,6:9].mean(), 6.9)

            # set from another image
            img2 = img.clone()
            img3 = img.clone()
            img[img2] = 0
            with self.assertRaises(Exception):
                # different physical space
                img2.set_direction(img3.direction*2)
                img3[img2] = 0

    def test__repr__(self):
        for img in self.imgs:
            s = img.__repr__()


class TestModule_ants_image(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16')).clone('float')
        img3d = ants.image_read(ants.get_ants_data('mni')).clone('float')
        self.imgs = [img2d, img3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_copy_image_info(self):
        for img in self.imgs:
            img2 = img.clone()
            img2.set_spacing([6.9]*img.dimension)
            img2.set_origin([6.9]*img.dimension)
            self.assertTrue(not ants.image_physical_space_consistency(img,img2))

            img3 = ants.copy_image_info(reference=img, target=img2)
            self.assertTrue(ants.image_physical_space_consistency(img,img3))

    def test_get_spacing(self):
        for img in self.imgs:
            spacing = ants.get_spacing(img)
            self.assertTrue(isinstance(spacing, tuple))
            self.assertEqual(len(ants.get_spacing(img)), img.dimension)

    def test_set_spacing(self):
        for img in self.imgs:
            # set spacing from list
            new_spacing_list = [6.9]*img.dimension
            ants.set_spacing(img, new_spacing_list)
            self.assertEqual(img.spacing, tuple(new_spacing_list))

            # set spacing from tuple
            new_spacing_tuple = tuple(new_spacing_list)
            ants.set_spacing(img, new_spacing_tuple)
            self.assertEqual(ants.get_spacing(img), new_spacing_tuple)

    def test_get_origin(self):
        for img in self.imgs:
            origin = ants.get_origin(img)
            self.assertTrue(isinstance(origin, tuple))
            self.assertEqual(len(ants.get_origin(img)), img.dimension)

    def test_set_origin(self):
        for img in self.imgs:
            # set spacing from list
            new_origin_list = [6.9]*img.dimension
            ants.set_origin(img, new_origin_list)
            self.assertEqual(img.origin, tuple(new_origin_list))

            # set spacing from tuple
            new_origin_tuple = tuple(new_origin_list)
            ants.set_origin(img, new_origin_tuple)
            self.assertEqual(ants.get_origin(img), new_origin_tuple)

    def test_get_direction(self):
        for img in self.imgs:
            direction = ants.get_direction(img)
            self.assertTrue(isinstance(direction,np.ndarray))
            self.assertTrue(ants.get_direction(img).shape, (img.dimension,img.dimension))

    def test_set_direction(self):
        for img in self.imgs:
            new_direction = np.eye(img.dimension)*3
            ants.set_direction(img, new_direction)
            nptest.assert_allclose(ants.get_direction(img), new_direction)

    def test_image_physical_spacing_consistency(self):
        for img in self.imgs:
            self.assertTrue(ants.image_physical_space_consistency(img,img))
            self.assertTrue(ants.image_physical_space_consistency(img,img,datatype=True))
            clonetype = 'float' if img.pixeltype != 'float' else 'unsigned int'
            img2 = img.clone(clonetype)
            self.assertTrue(ants.image_physical_space_consistency(img,img2))
            self.assertTrue(not ants.image_physical_space_consistency(img,img2,datatype=True))

            # test incorrectness #
            # bad spacing
            img2 = img.clone()
            img2.set_spacing([6.96]*img.dimension)
            self.assertTrue(not ants.image_physical_space_consistency(img,img2))

            # bad origin
            img2 = img.clone()
            img2.set_origin([6.96]*img.dimension)
            self.assertTrue(not ants.image_physical_space_consistency(img,img2))

            # bad direction
            img2 = img.clone()
            img2.set_direction(img.direction*2)
            self.assertTrue(not ants.image_physical_space_consistency(img,img2))

            # bad dimension
            ndim = img.dimension
            img2 = ants.from_numpy(np.random.randn(*tuple([69]*(ndim+1))).astype('float32'))
            self.assertTrue(not ants.image_physical_space_consistency(img,img2))

            # only one image
            with self.assertRaises(Exception):
                ants.image_physical_space_consistency(img)
            # not an ANTsImage
            with self.assertRaises(Exception):
                ants.image_physical_space_consistency(img, 12)

        # false because of components
        vecimg = ants.from_numpy(np.random.randn(69,70,3), has_components=True)
        vecimg2 = ants.from_numpy(np.random.randn(69,70,4), has_components=True)
        self.assertTrue(not ants.image_physical_space_consistency(vecimg, vecimg2, datatype=True))

    def test_image_type_cast(self):
        # test with list of images
        imgs2 = ants.image_type_cast(self.imgs)
        for img in imgs2:
            self.assertTrue(img.pixeltype, 'float')

        # not a list or tuple
        with self.assertRaises(Exception):
            ants.image_type_cast(self.imgs[0])

        # cast from unsigned int
        imgs = [i.clone() for i in self.imgs]
        imgs[0] = imgs[0].clone('unsigned int')
        imgs2 = ants.image_type_cast(imgs)


    def test_allclose(self):
        for img in self.imgs:
            img2 = img.clone()
            self.assertTrue(ants.allclose(img,img2))
            self.assertTrue(ants.allclose(img*6.9, img2*6.9))
            self.assertTrue(not ants.allclose(img, img2*6.9))

    def test_pickle(self):
        import ants
        import pickle
        img = ants.image_read( ants.get_ants_data("mni"))
        img_pickled = pickle.dumps(img)
        img2 = pickle.loads(img_pickled)

        self.assertTrue(ants.allclose(img, img2))
        self.assertTrue(ants.image_physical_space_consistency(img, img2))

        img = ants.image_read( ants.get_ants_data("r16"))
        img_pickled = pickle.dumps(img)
        img2 = pickle.loads(img_pickled)

        self.assertTrue(ants.allclose(img, img2))
        self.assertTrue(ants.image_physical_space_consistency(img, img2))


if __name__ == '__main__':
    run_tests()
