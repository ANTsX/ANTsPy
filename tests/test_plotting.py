"""
Test ants.learn module

nptest.assert_allclose
self.assertEqual
self.assertTrue
"""

import os
import unittest
from common import run_tests
from tempfile import mktemp

# Prevent displaying figures
import matplotlib as mpl
backend_ =  mpl.get_backend() 
mpl.use("Agg")  

import numpy as np
import numpy.testing as nptest

import ants


class TestModule_plot(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        for img in self.imgs:
            ants.plot(img)
            ants.plot(img, overlay=img*2)
            ants.plot(img, overlay=img*2)
            ants.plot(img, filename=filename)
    
    def test_extra_plot(self):
        img = ants.image_read(ants.get_ants_data('r16'))
        ants.plot(img, overlay=img*2, domain_image_map=ants.image_read(ants.get_data('r64')))

        img = ants.image_read(ants.get_ants_data('r16'))
        ants.plot(img, crop=True)

        img = ants.image_read(ants.get_ants_data('mni'))
        ants.plot(img, overlay=img*2, 
                  domain_image_map=ants.image_read(ants.get_data('mni')).resample_image((4,4,4)))

        img = ants.image_read(ants.get_ants_data('mni'))
        ants.plot(img, overlay=img*2, reorient=True, crop=True)
        
    def test_random(self):
        img = ants.image_read(ants.get_ants_data('r16'))
        img3 = ants.image_read(ants.get_data('r64'))
        img2 = ants.image_read(ants.get_ants_data('mni'))
        imgv = ants.merge_channels([img2])
        
        ants.plot(img2, axis='x', scale=True, ncol=1)
        ants.plot(img2, axis='y', scale=(0.05, 0.95))
        ants.plot(img2, axis='z', slices=[10,20,30], title='Test', cbar=True,
                  cbar_vertical=True)
        ants.plot(img2, cbar=True, cbar_vertical=False)
        ants.plot(img, black_bg=False, title='Test', cbar=True)
        imgx = img2.clone()
        imgx.set_spacing((10,1,1))
        ants.plot(imgx)
        ants.plot(ants.get_ants_data('r16'), overlay=ants.get_data('r64'), blend=True)
        with self.assertRaises(Exception):
            ants.plot(ants.get_ants_data('r16'), overlay=123)
        with self.assertRaises(Exception):
            ants.plot(ants.get_ants_data('r16'), overlay=ants.merge_channels([img,img]))
        ants.plot(ants.from_numpy(np.zeros((100,100))))
        ants.plot(img.clone('unsigned int'))
        
        ants.plot(img, domain_image_map=img3)
        
        with self.assertRaises(Exception):
            ants.plot(123)
        
        
        
        
        
        
        
class TestModule_plot_ortho(unittest.TestCase):

    def setUp(self):
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img3d]

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        for img in self.imgs:
            ants.plot_ortho(img)
            ants.plot_ortho(img, filename=filename)
            
    def test_plot_extra(self):
        img = ants.image_read(ants.get_ants_data('mni'))
        ants.plot_ortho(img, overlay=img*2, 
                  domain_image_map=ants.image_read(ants.get_data('mni')))

        img = ants.image_read(ants.get_ants_data('mni'))
        ants.plot_ortho(img, overlay=img*2, reorient=True, crop=True)
        
    def test_random_params(self):
        img = ants.image_read(ants.get_ants_data('mni')).resample_image((4,4,4))
        img2 = ants.image_read(ants.get_data('r16'))
        ants.plot_ortho(ants.get_data('mni'), overlay=ants.get_data('mni'))
        with self.assertRaises(Exception):
            ants.plot_ortho(123)
        with self.assertRaises(Exception):
            ants.plot_ortho(img2)
        with self.assertRaises(Exception):
            ants.plot_ortho(img, overlay=img2)
            
        imgx = img.clone()
        imgx.set_spacing((3,3,3))
        ants.plot_ortho(img,overlay=imgx)
        ants.plot_ortho(img.clone('unsigned int'),overlay=img, blend=True)
        
        imgx = img.clone()
        imgx.set_spacing((10,1,1))
        ants.plot_ortho(imgx)
        
        ants.plot_ortho(img, flat=True, title='Test', text='This is a test')
        ants.plot_ortho(img, title='Test', text='This is a test', cbar=True)
        
        with self.assertRaises(Exception):
            ants.plot_ortho(img, domain_image_map=123)
            
        with self.assertRaises(Exception):
            ants.plot_orto(ants.merge_channels([img,img]))
        

class TestModule_plot_ortho_stack(unittest.TestCase):

    def setUp(self):
        self.img = ants.image_read(ants.get_ants_data('mni'))

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        ants.plot_ortho_stack([self.img, self.img])
        ants.plot_ortho_stack([self.img, self.img], filename=filename)
        
    def test_extra_ortho_stack(self):
        img = ants.image_read(ants.get_ants_data('mni'))
        ants.plot_ortho_stack([img, img], overlays=[img*2, img*2],
                  domain_image_map=ants.image_read(ants.get_data('mni')))

        img = ants.image_read(ants.get_ants_data('mni'))
        ants.plot_ortho_stack([img, img], overlays=[img*2, img*2],  reorient=True, crop=True)

class TestModule_plot_hist(unittest.TestCase):

    def setUp(self):
        img2d = ants.image_read(ants.get_ants_data('r16'))
        img3d = ants.image_read(ants.get_ants_data('mni'))
        self.imgs = [img2d, img3d]

    def tearDown(self):
        pass

    def test_plot_example(self):
        filename = mktemp(suffix='.png')
        for img in self.imgs:
            ants.plot_hist(img)

class TestModule_plot_grid(unittest.TestCase):

    def setUp(self):
        mni1 = ants.image_read(ants.get_data('mni'))
        mni2 = mni1.smooth_image(1.)
        mni3 = mni1.smooth_image(2.)
        mni4 = mni1.smooth_image(3.)
        self.images3d = np.asarray([[mni1, mni2],
                                    [mni3, mni4]])
        self.images2d = np.asarray([[mni1.slice_image(2,100), mni2.slice_image(2,100)],
                                    [mni3.slice_image(2,100), mni4.slice_image(2,100)]])

    def tearDown(self):
        pass

    def test_plot_example(self):
        ants.plot_grid(self.images3d, slices=100)
        # should take middle slices if none are given
        ants.plot_grid(self.images3d)
        # should work with 2d images
        ants.plot_grid(self.images2d)
        
    def test_examples(self):
        mni1 = ants.image_read(ants.get_data('mni'))
        mni2 = mni1.smooth_image(1.)
        mni3 = mni1.smooth_image(2.)
        mni4 = mni1.smooth_image(3.)
        images = np.asarray([[mni1, mni2],
                            [mni3, mni4]])
        slices = np.asarray([[100, 100],
                            [100, 100]])
        ants.plot_grid(images=images, slices=slices, title='2x2 Grid')
        images2d = np.asarray([[mni1.slice_image(2,100), mni2.slice_image(2,100)],
                            [mni3.slice_image(2,100), mni4.slice_image(2,100)]])
        ants.plot_grid(images=images2d, title='2x2 Grid Pre-Sliced')
        ants.plot_grid(images.reshape(1,4), slices.reshape(1,4), title='1x4 Grid')
        ants.plot_grid(images.reshape(4,1), slices.reshape(4,1), title='4x1 Grid')

        # Padding between rows and/or columns
        ants.plot_grid(images, slices, cpad=0.02, title='Col Padding')
        ants.plot_grid(images, slices, rpad=0.02, title='Row Padding')
        ants.plot_grid(images, slices, rpad=0.02, cpad=0.02, title='Row and Col Padding')

        # Adding plain row and/or column labels
        ants.plot_grid(images, slices, title='Adding Row Labels', rlabels=['Row #1', 'Row #2'])
        ants.plot_grid(images, slices, title='Adding Col Labels', clabels=['Col #1', 'Col #2'])
        ants.plot_grid(images, slices, title='Row and Col Labels',
                    rlabels=['Row 1', 'Row 2'], clabels=['Col 1', 'Col 2'])

        # Making a publication-quality image
        images = np.asarray([[mni1, mni2, mni2],
                            [mni3, mni4, mni4]])
        slices = np.asarray([[100, 100, 100],
                            [100, 100, 100]])
        axes = np.asarray([[0, 1, 2],
                        [0, 1, 2]])
        ants.plot_grid(images, slices, axes, title='Publication Figures with ANTsPy',
                    tfontsize=20, title_dy=0.03, title_dx=-0.04,
                    rlabels=['Row 1', 'Row 2'],
                    clabels=['Col 1', 'Col 2', 'Col 3'],
                    rfontsize=16, cfontsize=16)


class TestModule_plot_ortho_stack(unittest.TestCase):

    def setUp(self):
        pass
    def tearDown(self):
        pass
    
    def test_random_ortho_stack_params(self):
        img = ants.image_read(ants.get_data('mni')).resample_image((4,4,4))
        img2 = ants.image_read(ants.get_data('r16')).resample_image((4,4))
        
        ants.plot_ortho_stack([ants.get_data('mni'), ants.get_data('mni')])
        ants.plot_ortho_stack([ants.get_data('mni'), ants.get_data('mni')],
                              overlays=[ants.get_data('mni'), ants.get_data('mni')])
        
        with self.assertRaises(Exception):
            ants.plot_ortho_stack([1,2,3])
        with self.assertRaises(Exception):
            ants.plot_ortho_stack([img2,img2])
        with self.assertRaises(Exception):
            ants.plot_ortho_stack([img,img], overlays=[img2,img2])
        with self.assertRaises(Exception):
            ants.plot_ortho_stack([img,img], overlays=[1,2])
            
        imgx = img.clone()
        imgx.set_spacing((2,2,2))
        ants.plot_ortho_stack([img,imgx])
        
        imgx.set_spacing((2,1,1))
        ants.plot_ortho_stack([imgx,img])
        
        ants.plot_ortho_stack([img,img], scale=True, transpose=True,
                              title='Test', colpad=1, rowpad=1,
                              xyz_lines=True)
        ants.plot_ortho_stack([img,img], scale=(0.05,0.95))
            
if __name__ == '__main__':
    run_tests()
