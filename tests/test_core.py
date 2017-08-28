"""
Test Core Module

Pixel Types
-----------
(unsigned) char
(unsigned) short
(unsigned) int
float
double
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


class TestBiasCorrection(unittest.TestCase):

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

    def test_n3_bias_field_correction(self): 
        for img in self.imgs:
            for downsample_factor in range(3,4,5):
                if img.dimension == 2:
                    img_n3 = ants.n3_bias_field_correction(img, downsample_factor)

                    # images should not be the same after bias correction
                    self.assertNotEqual(img.numpy().sum(), img_n3.numpy().sum())

                    # but they should have the same properties
                    self.assertEqual(img.pixeltype, img_n3.pixeltype)
                    self.assertEqual(img.shape, img_n3.shape)
                    self.assertEqual(img.dimension, img_n3.dimension)

                    self.assertEqual(img.spacing, img_n3.spacing)
                    self.assertEqual(img.origin, img_n3.origin)
                    nptest.assert_allclose(img.direction, img_n3.direction)



class TestANTsImageClass(unittest.TestCase):
    """
    Test ants.ANTsImage class
    """
    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        img_paths = []
        for dtype in ['CHAR', 'DOUBLE', 'FLOAT', 'INT', 'SHORT', 
                      'UNSIGNEDCHAR', 'UNSIGNEDINT', 'UNSIGNEDSHORT']:
            for dim in [2,3]:
                img_paths.append('image_%s_%iD.nii.gz' % (dtype, dim))

        self.img_paths = [os.path.join(self.datadir, img_path) for img_path in img_paths]

        self.imgs = [ants.image_read(ip) for ip in self.img_paths]

        self.NUM_ARITH_TESTS = 2

    def tearDown(self):
        pass

    def test_get_and_set_origin(self):
        """
        test get and set of origin
        """
        for img_path in self.img_paths:
            ndim = _parse_dim_from_string(img_path)
            img = ants.image_read(img_path)

            origin = img.origin
            # assert that there is one origin value for each dimension
            self.assertEqual(len(origin), ndim)

            # try to set neworigin
            img.set_origin(tuple([3.6]*ndim))
            new_origin = img.origin

            self.assertEqual(new_origin, tuple([3.6]*ndim))

    def test_get_and_set_spacing(self):
        """
        test get and set of spacing
        """
        for img_path in self.img_paths:
            ndim = _parse_dim_from_string(img_path)
            img = ants.image_read(img_path)

            spacing = img.spacing
            # assert that there is one spacing value for each dimension
            self.assertEqual(len(spacing), ndim)

            # try to set neworigin
            img.set_spacing(tuple([3.6]*ndim))
            new_spacing = img.spacing

            self.assertEqual(new_spacing, tuple([3.6]*ndim))

    def test_get_and_set_direction(self):
        for img_path in self.img_paths:
            ndim = _parse_dim_from_string(img_path)
            img = ants.image_read(img_path)

            direction = img.direction

            self.assertTrue(isinstance(direction, np.ndarray))
            self.assertEqual(direction.shape, (ndim, ndim))

            # try to set neworigin
            set_direction = np.eye(ndim)*3.6
            img.set_direction(set_direction)
            get_direction = img.direction

            nptest.assert_allclose(set_direction, get_direction)

    def test_image_clone(self):
        for img in self.imgs:
            img_clone = img.clone()

            array1 = img.numpy()
            array2 = img_clone.numpy()

            # test data
            nptest.assert_allclose(array1, array2)

            # test properties
            self.assertEqual(img.dimension, img_clone.dimension)
            self.assertEqual(img.pixeltype, img_clone.pixeltype)
            self.assertEqual(img.components, img_clone.components)
            self.assertEqual(img.shape, img_clone.shape)

            # test physical space
            self.assertEqual(img.origin, img_clone.origin)
            self.assertEqual(img.spacing, img_clone.spacing)
            nptest.assert_allclose(img.direction, img_clone.direction)

    def test_image_cast(self):
        for img in self.imgs:
            for dtype in ['uint8', 'int8',
                          'uint16', 'int16',
                          'uint32', 'int32',
                          'float32', 'float64']:
                img_cast = img.cast(dtype)

                array1 = img.numpy().astype(dtype)
                array2 = img_cast.numpy()

                # test data
                nptest.assert_allclose(array1, array2)

                # test properties
                self.assertEqual(img.dimension, img_cast.dimension)
                self.assertEqual(dtype, img_cast.datatype)
                self.assertEqual(img.components, img_cast.components)
                self.assertEqual(img.shape, img_cast.shape)

                # test physical space
                self.assertEqual(img.origin, img_cast.origin)
                self.assertEqual(img.spacing, img_cast.spacing)
                nptest.assert_allclose(img.direction, img_cast.direction)

    def test_new_image_like(self):
        for img in self.imgs:
            new_data = np.random.randn(*img.shape).astype(img.datatype)
            img_new = img.new_image_like(new_data)

            array1 = img.numpy()
            array2 = img_new.numpy()

            # data should NOT be equal
            self.assertNotEqual(array1.sum(), array2.sum())

            # test properties
            self.assertEqual(img.dimension, img_new.dimension)
            self.assertEqual(img.pixeltype, img_new.pixeltype)
            self.assertEqual(img.components, img_new.components)
            self.assertEqual(img.shape, img_new.shape)

            # test physical space
            self.assertEqual(img.origin, img_new.origin)
            self.assertEqual(img.spacing, img_new.spacing)
            nptest.assert_allclose(img.direction, img_new.direction)        


    def test_add_image_to_image(self):
        for img1 in self.imgs:
            for idx, img2 in enumerate(self.imgs):
                if idx < self.NUM_ARITH_TESTS:
                    ndim1 = img1.dimension
                    ndim2 = img2.dimension
                    if (ndim1 == ndim2):
                        p1 = img1.datatype
                        p2 = img2.datatype
                        p3 = _get_larger_dtype(p1, p2)
                        
                        img3 = img1 + img2

                        arr1 = img1.numpy()
                        arr2 = img2.numpy()
                        arr3 = img3.numpy()

                        nptest.assert_allclose(arr1 + arr2, arr3)

    def test_subtract_image_from_image(self):
        for img1 in self.imgs:
            for idx, img2 in enumerate(self.imgs):
                if idx < self.NUM_ARITH_TESTS:
                    ndim1 = img1.dimension
                    ndim2 = img2.dimension
                    if (ndim1 == ndim2):
                        p1 = img1.datatype
                        p2 = img2.datatype
                        p3 = _get_larger_dtype(p1, p2)
                        
                        img3 = img1 - img2

                        arr1 = img1.numpy()
                        arr2 = img2.numpy()
                        arr3 = img3.numpy()
                        nptest.assert_allclose(arr1 - arr2, arr3)

    def test_divide_image_by_image(self):
        for img1 in self.imgs:
            for idx, img2 in enumerate(self.imgs):
                if idx < self.NUM_ARITH_TESTS:
                    ndim1 = img1.dimension
                    ndim2 = img2.dimension
                    if (ndim1 == ndim2) and ('char' not in img1.pixeltype) and ('char' not in img2.pixeltype):
                        p1 = img1.datatype
                        p2 = img2.datatype
                        p3 = _get_larger_dtype(p1, p2)
                        img3 = img1 / img2

                        arr1 = img1.numpy().astype(p3)
                        arr2 = img2.numpy().astype(p3)
                        arr3 = img3.numpy()

                        nptest.assert_allclose(arr1 / arr2, arr3)

    def test_multiply_image_by_image(self):
        for img1 in self.imgs:
            for idx, img2 in enumerate(self.imgs):
                if idx < self.NUM_ARITH_TESTS:
                    ndim1 = img1.dimension
                    ndim2 = img2.dimension
                    if (ndim1 == ndim2):
                        p1 = img1.datatype
                        p2 = img2.datatype
                        p3 = _get_larger_dtype(p1, p2)
                        
                        img3 = img1 * img2

                        arr1 = img1.numpy()
                        arr2 = img2.numpy()
                        arr3 = img3.numpy()

                        nptest.assert_allclose(arr1 * arr2, arr3)


class TestCoreIO(unittest.TestCase):
    """
    Test `ants.core` io operations
    """

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        img_paths = []
        for dtype in ['CHAR', 'DOUBLE', 'FLOAT', 'INT', 'SHORT', 
                      'UNSIGNEDCHAR', 'UNSIGNEDINT', 'UNSIGNEDSHORT']:
            for dim in [2,3]:
                img_paths.append('image_%s_%iD.nii.gz' % (dtype, dim))

        self.img_paths = [os.path.join(self.datadir, img_path) for img_path in img_paths]

    def tearDown(self):
        pass

    def test_infer_image_info(self):
        """
        infer the pixeltype, dimension, and component info from an image file
        """
        for img_path in self.img_paths:
            img_info = ants.infer_image_info(img_path)

            ptype = _parse_pixeltype_from_string(img_path)
            dim = _parse_dim_from_string(img_path)
            self.assertEqual(img_info, (ptype, dim, 'scalar'))


    def test_image_read(self):
        """
        reading images from file
        """
        for img_path in self.img_paths:
            img = ants.image_read(img_path)

            ptype = _parse_pixeltype_from_string(img_path)
            dim = _parse_dim_from_string(img_path)
            self.assertEqual(img.pixeltype, ptype)
            self.assertEqual(img.dimension, dim)
            self.assertEqual(img.components, 1)

    def test_image_read_then_to_numpy(self):
        """
        convert ANTsImage's to numpy arrays
        """
        for img_path in self.img_paths:
            img = ants.image_read(img_path)
            img_array = img.numpy()

            npy_dtype = _parse_numpy_pixeltype_from_string(img_path)
            dim = _parse_dim_from_string(img_path)

            self.assertGreater(img_array.sum(), 0)
            self.assertGreater(len(img_array), 0)
            self.assertEqual(img_array.dtype, npy_dtype)
            self.assertEqual(img_array.ndim, dim)
            self.assertEqual(tuple(img_array.shape), tuple(img.shape))

    def test_image_read_then_to_numpy_WrapITK_equality(self):
        """
        Test that reading an image then converting it to numpy
        is the same in ANTsPy and the python version of ITK
        """
        for img_path in self.img_paths:
            # my itk doesnt wrap integer or unsigned integer...
            if 'INT' not in img_path:
                ants_array = ants.image_read(img_path).numpy()
                itk_array = itk.GetArrayViewFromImage(itk.imread(img_path))
                self.assertEqual(ants_array.sum(), itk_array.sum())
                self.assertEqual(ants_array.dtype, itk_array.dtype)


    def test_image_write(self):
        for img_path in self.img_paths:
            img = ants.image_read(img_path)
            new_path = img_path.replace('.nii.gz', '_write.nii.gz')
            ants.image_write(img, new_path)
            new_img = ants.image_read(new_path)

            self.assertEqual(img.pixeltype, new_img.pixeltype)
            self.assertEqual(img.dimension, new_img.pixeltype)
            self.assertEqual(img.components, new_img.components)

            self.assertEqual(img.spacing, new_img.spacing)
            self.assertEqual(img.origin, new_img.origin)
            nptest.assertEqual(img.direction, new_img.direction)

            nptest.assert_allclose(img.numpy(), new_img.numpy())

    def test_image_to_file(self):
        for img_path in self.img_paths:
            img = ants.image_read(img_path)
            new_path = img_path.replace('.nii.gz', '_write.nii.gz')
            img.to_file(new_path)
            new_img = ants.image_read(new_path)

            self.assertEqual(img.pixeltype, new_img.pixeltype)
            self.assertEqual(img.dimension, new_img.dimension)
            self.assertEqual(img.components, new_img.components)

            self.assertEqual(img.spacing, new_img.spacing)
            self.assertEqual(img.origin, new_img.origin)
            nptest.assertEqual(img.direction, new_img.direction)
            
            nptest.assert_allclose(img.numpy(), new_img.numpy())


    def test_from_numpy_without_image_info(self):
        for shape in [(20,20), (20,20,20)]:
            for dtype in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64']:
                ndim = len(shape)
                array = np.random.randint(0,254, shape).astype(dtype)

                spacing = tuple([3.6]*ndim)
                origin = tuple([6.9]*ndim)
                direction = np.eye(ndim)*9.6

                image = ants.from_numpy(array)

                img_array = image.numpy()

                nptest.assert_allclose(array, img_array)
                self.assertEqual(image.dimension, ndim)
                self.assertEqual(image.shape, shape)

    def test_from_numpy_with_image_info(self):
        for shape in [(20,20), (20,20,20)]:
            for dtype in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64']:
                ndim = len(shape)
                array = np.random.randint(0,254, shape).astype(dtype)

                spacing = tuple([3.6]*ndim)
                origin = tuple([6.9]*ndim)
                direction = np.eye(ndim)*9.6

                image = ants.from_numpy(array, spacing=spacing, origin=origin, direction=direction)

                img_array = image.numpy()

                nptest.assert_allclose(array.sum(), img_array.sum())
                self.assertEqual(image.dimension, ndim)
                self.assertEqual(image.shape, shape)
                self.assertEqual(image.spacing, spacing)
                self.assertEqual(image.origin, origin)
                nptest.assert_allclose(image.direction, direction)


if __name__ == '__main__':
    run_tests()
