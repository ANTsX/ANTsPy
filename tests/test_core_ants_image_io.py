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

class TestModule_ants_image_io(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16')).clone('float')
        img3d = ants.image_read(ants.get_ants_data('mni')).clone('float')
        arr2d = np.random.randn(69,70).astype('float32')
        arr3d = np.random.randn(69,70,71).astype('float32')
        vecimg2d = ants.from_numpy(np.random.randn(69,70,4), has_components=True)
        vecimg3d = ants.from_numpy(np.random.randn(69,70,71,2), has_components=True)
        self.imgs = [img2d, img3d]
        self.arrs = [arr2d, arr3d]
        self.vecimgs = [vecimg2d, vecimg3d]
        self.pixeltypes = ['unsigned char', 'unsigned int', 'float']

    def tearDown(self):
        pass

    def test_from_numpy(self):
        self.setUp()

        # no physical space info
        for arr in self.arrs:
            img = ants.from_numpy(arr)

            self.assertTrue(img.dimension, arr.ndim)
            self.assertTrue(img.shape, arr.shape)
            self.assertTrue(img.dtype, arr.dtype.name)
            nptest.assert_allclose(img.numpy(), arr)

            new_origin = tuple([6.9]*arr.ndim)
            new_spacing = tuple([3.6]*arr.ndim)
            new_direction = np.eye(arr.ndim)*9.6
            img2 = ants.from_numpy(arr, origin=new_origin, spacing=new_spacing, direction=new_direction)

            self.assertEqual(img2.origin, new_origin)
            self.assertEqual(img2.spacing, new_spacing)
            nptest.assert_allclose(img2.direction, new_direction)

        # test with components
        arr2d_components = np.random.randn(69,70,4).astype('float32')
        img = ants.from_numpy(arr2d_components, has_components=True)
        self.assertEqual(img.components, arr2d_components.shape[-1])
        nptest.assert_allclose(arr2d_components, img.numpy())

    def test_make_image(self):
        self.setUp()

        for arr in self.arrs:
            voxval = 6.
            img = ants.make_image(arr.shape, voxval=voxval)
            self.assertTrue(img.dimension, arr.ndim)
            self.assertTrue(img.shape, arr.shape)
            nptest.assert_allclose(img.mean(), voxval)

            new_origin = tuple([6.9]*arr.ndim)
            new_spacing = tuple([3.6]*arr.ndim)
            new_direction = np.eye(arr.ndim)*9.6
            img2 = ants.make_image(arr.shape, voxval=voxval, origin=new_origin, spacing=new_spacing, direction=new_direction)

            self.assertTrue(img2.dimension, arr.ndim)
            self.assertTrue(img2.shape, arr.shape)
            nptest.assert_allclose(img2.mean(), voxval)
            self.assertEqual(img2.origin, new_origin)
            self.assertEqual(img2.spacing, new_spacing)
            nptest.assert_allclose(img2.direction, new_direction)

            for ptype in self.pixeltypes:
                img = ants.make_image(arr.shape, voxval=1., pixeltype=ptype)
                self.assertEqual(img.pixeltype, ptype)

        # test with components
        img = ants.make_image((69,70,4), has_components=True)
        self.assertEqual(img.components, 4)
        self.assertEqual(img.dimension, 2)
        nptest.assert_allclose(img.mean(), 0.)

        img = ants.make_image((69,70,71,4), has_components=True)
        self.assertEqual(img.components, 4)
        self.assertEqual(img.dimension, 3)
        nptest.assert_allclose(img.mean(), 0.)

        # set from image
        for img in self.imgs:
            mask = ants.image_clone(  img > img.mean(), pixeltype = 'float' )
            arr = img[mask]
            img2 = ants.make_image(mask, voxval=arr)
            nptest.assert_allclose(img2.numpy(), (img*mask).numpy())
            self.assertTrue(ants.image_physical_space_consistency(img2,mask))

            # set with arr.ndim > 1
            img2 = ants.make_image(mask, voxval=np.expand_dims(arr,-1))
            nptest.assert_allclose(img2.numpy(), (img*mask).numpy())
            self.assertTrue(ants.image_physical_space_consistency(img2,mask))

            #with self.assertRaises(Exception):
            #    # wrong number of non-zero voxels
            #    img3 = ants.make_image(img, voxval=arr)


    def test_matrix_to_images(self):
        # def matrix_to_images(data_matrix, mask):
        for img in self.imgs:
            imgmask = ants.image_clone(  img > img.mean(), pixeltype = 'float' )
            data = img[imgmask]
            dataflat = data.reshape(1,-1)
            mat = np.vstack([dataflat,dataflat]).astype('float32')
            imglist = ants.matrix_to_images(mat, imgmask)
            nptest.assert_allclose((img*imgmask).numpy(), imglist[0].numpy())
            nptest.assert_allclose((img*imgmask).numpy(), imglist[1].numpy())
            self.assertTrue(ants.image_physical_space_consistency(img,imglist[0]))
            self.assertTrue(ants.image_physical_space_consistency(img,imglist[1]))

            # go back to matrix
            mat2 = ants.images_to_matrix(imglist, imgmask)
            nptest.assert_allclose(mat, mat2)

            # test with matrix.ndim > 2
            img = img.clone()
            img.set_direction(img.direction*2)
            imgmask = ants.image_clone(  img > img.mean(), pixeltype = 'float' )
            arr = (img*imgmask).numpy()
            arr = arr[arr>=0.5]
            arr2 = arr.copy()
            mat = np.stack([arr,arr2])
            imglist = ants.matrix_to_images(mat, imgmask)
            for im in imglist:
                self.assertTrue(ants.allclose(im, imgmask*img))
                self.assertTrue(ants.image_physical_space_consistency(im, imgmask))

            # test for wrong number of voxels
            #with self.assertRaises(Exception):
            #    arr = (img*imgmask).numpy()
            #    arr = arr[arr>0.5]
            #    arr2 = arr.copy()
            #    mat = np.stack([arr,arr2])
            #    imglist = ants.matrix_to_images(mat, img)

    def test_images_to_matrix(self):
        # def images_to_matrix(image_list, mask=None, sigma=None, epsilon=0):
        for img in self.imgs:
            mask = ants.image_clone(  img > img.mean(), pixeltype = 'float' )
            imglist = [img.clone(),img.clone(),img.clone()]
            imgmat = ants.images_to_matrix(imglist, mask=mask)
            self.assertTrue(imgmat.shape[0] == len(imglist))
            self.assertTrue(imgmat.shape[1] == (mask>0).sum())
            self.assertTrue(np.allclose(img[mask], imgmat[0,:]))

            # go back to images
            imglist2 = ants.matrix_to_images(imgmat, mask)
            for i1,i2 in zip(imglist,imglist2):
                self.assertTrue(ants.image_physical_space_consistency(i1,i2))
                nptest.assert_allclose(i1.numpy()*mask.numpy(),i2.numpy())

            if img.dimension == 2:
                # with sigma
                mask = ants.image_clone(  img > img.mean(), pixeltype = 'float' )
                imglist = [img.clone(),img.clone(),img.clone()]
                imgmat = ants.images_to_matrix(imglist, mask=mask, sigma=2.)

                # with no mask
                imgmat = ants.images_to_matrix(imglist)

                # Mask not binary
                mask = ants.image_clone(  img / img.mean(), pixeltype = 'float' )
                imgmat = ants.images_to_matrix(imglist, mask=mask, epsilon=1)

                # with mask of different shape
                s = [65]*img.dimension
                mask2 = ants.from_numpy(np.random.randn(*s), spacing=[4.0, 4.0])
                mask2 = mask2 > mask2.mean()
                imgmat = ants.images_to_matrix(imglist, mask=mask2)
                self.assertTrue(imgmat.shape[0] == len(imglist))
                self.assertTrue(imgmat.shape[1] == (mask2>0).sum())



    def timeseries_to_matrix(self):
        img = ants.make_image( (10,10,10,5 ) )
        mat = ants.timeseries_to_matrix( img )

        img = ants.make_image( (10,10,10,5 ) )
        mask = ants.ndimage_to_list( img )[0] * 0
        mask[ 4:8, 4:8, 4:8 ] = 1
        mat = ants.timeseries_to_matrix( img, mask = mask )
        img2 = ants.matrix_to_timeseries( img,  mat, mask)

    def test_image_header_info(self):
        # def image_header_info(filename):
        for img in self.imgs:
            img.set_spacing([6.9]*img.dimension)
            img.set_origin([3.6]*img.dimension)
            tmpfile = mktemp(suffix='.nii.gz')
            ants.image_write(img, tmpfile)

            info = ants.image_header_info(tmpfile)
            self.assertEqual(info['dimensions'], img.shape)
            nptest.assert_allclose(info['direction'], img.direction)
            self.assertEqual(info['nComponents'], img.components)
            self.assertEqual(info['nDimensions'], img.dimension)
            self.assertEqual(info['origin'], img.origin)
            self.assertEqual(info['pixeltype'], img.pixeltype)
            self.assertEqual(info['pixelclass'], 'vector' if img.has_components else 'scalar')
            self.assertEqual(info['spacing'], img.spacing)

            try:
                os.remove(tmpfile)
            except:
                pass

        # test on vector image
        img = ants.from_numpy(np.random.randn(69,60,4).astype('float32'), has_components=True)
        tmpfile = mktemp(suffix='.nii.gz')
        ants.image_write(img, tmpfile)
        info = ants.image_header_info(tmpfile)
        self.assertEqual(info['dimensions'], img.shape)
        nptest.assert_allclose(info['direction'], img.direction)
        self.assertEqual(info['nComponents'], img.components)
        self.assertEqual(info['nDimensions'], img.dimension)
        self.assertEqual(info['origin'], img.origin)
        self.assertEqual(info['pixeltype'], img.pixeltype)
        self.assertEqual(info['pixelclass'], 'vector' if img.has_components else 'scalar')
        self.assertEqual(info['spacing'], img.spacing)


        img = ants.from_numpy(np.random.randn(69,60,70,2).astype('float32'), has_components=True)
        tmpfile = mktemp(suffix='.nii.gz')
        ants.image_write(img, tmpfile)
        info = ants.image_header_info(tmpfile)
        self.assertEqual(info['dimensions'], img.shape)
        nptest.assert_allclose(info['direction'], img.direction)
        self.assertEqual(info['nComponents'], img.components)
        self.assertEqual(info['nDimensions'], img.dimension)
        self.assertEqual(info['origin'], img.origin)
        self.assertEqual(info['pixeltype'], img.pixeltype)
        self.assertEqual(info['pixelclass'], 'vector' if img.has_components else 'scalar')
        self.assertEqual(info['spacing'], img.spacing)

        # non-existant file
        with self.assertRaises(Exception):
            tmpfile = mktemp(suffix='.nii.gz')
            ants.image_header_info(tmpfile)

    def test_image_clone(self):
        for img in self.imgs:
            img = ants.image_clone(img, 'unsigned char')
            orig_ptype = img.pixeltype
            for ptype in self.pixeltypes:
                imgcloned = ants.image_clone(img, ptype)
                self.assertTrue(ants.image_physical_space_consistency(img,imgcloned))
                nptest.assert_allclose(img.numpy(), imgcloned.numpy())
                self.assertEqual(imgcloned.pixeltype, ptype)
                self.assertEqual(img.pixeltype, orig_ptype)

        for img in self.vecimgs:
            img = img.clone('unsigned char')
            orig_ptype = img.pixeltype
            for ptype in self.pixeltypes:
                imgcloned = ants.image_clone(img, ptype)
                self.assertTrue(ants.image_physical_space_consistency(img,imgcloned))
                self.assertEqual(imgcloned.components, img.components)
                nptest.assert_allclose(img.numpy(), imgcloned.numpy())
                self.assertEqual(imgcloned.pixeltype, ptype)
                self.assertEqual(img.pixeltype, orig_ptype)

    def test_image_read_write(self):
        # def image_read(filename, dimension=None, pixeltype='float'):
        # def image_write(image, filename):

        # test scalar images
        for img in self.imgs:
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255.
            img = img.clone('unsigned char')
            for ptype in self.pixeltypes:
                img = img.clone(ptype)
                tmpfile = mktemp(suffix='.nii.gz')
                ants.image_write(img, tmpfile)

                img2 = ants.image_read(tmpfile)
                self.assertTrue(ants.image_physical_space_consistency(img,img2))
                self.assertEqual(img2.components, img.components)
                nptest.assert_allclose(img.numpy(), img2.numpy())

            # unsupported ptype
            with self.assertRaises(Exception):
                ants.image_read(tmpfile, pixeltype='not-suppoted-ptype')

        # test vector images
        for img in self.vecimgs:
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255.
            img = img.clone('unsigned char')
            for ptype in self.pixeltypes:
                img = img.clone(ptype)
                tmpfile = mktemp(suffix='.nii.gz')
                ants.image_write(img, tmpfile)

                img2 = ants.image_read(tmpfile)
                self.assertTrue(ants.image_physical_space_consistency(img,img2))
                self.assertEqual(img2.components, img.components)
                nptest.assert_allclose(img.numpy(), img2.numpy())

        # test saving/loading as npy
        for img in self.imgs:
            tmpfile = mktemp(suffix='.npy')
            ants.image_write(img, tmpfile)
            img2 = ants.image_read(tmpfile)

            self.assertTrue(ants.image_physical_space_consistency(img,img2))
            self.assertEqual(img2.components, img.components)
            nptest.assert_allclose(img.numpy(), img2.numpy())

            # with no json header
            arr = img.numpy()
            tmpfile = mktemp(suffix='.npy')
            np.save(tmpfile, arr)
            img2 = ants.image_read(tmpfile)
            nptest.assert_allclose(img.numpy(), img2.numpy())

        # non-existant file
        with self.assertRaises(Exception):
            tmpfile = mktemp(suffix='.nii.gz')
            ants.image_read(tmpfile)


if __name__ == '__main__':
    run_tests()
