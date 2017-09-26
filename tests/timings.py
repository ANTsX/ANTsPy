"""
Timings against numpy/itk/nibabel/etc where appropriate
"""

import os

import nibabel as nib
import itk
import ants

import time

def time_nifti_to_numpy(N_TRIALS):
    """
    Times how fast a framework can read a nifti file and convert it to numpy
    """
    datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    img_paths = []
    for dtype in ['DOUBLE', 'FLOAT',
                  'UNSIGNEDCHAR',  'UNSIGNEDINT']:
        for dim in [2,3]:
            img_paths.append(os.path.join(datadir, 'image_%s_%iD.nii.gz' % (dtype, dim)))
    
    def test_nibabel():
        for img_path in img_paths:
            array = nib.load(img_path).get_data()

    def test_itk():
        for img_path in img_paths:
            array = itk.GetArrayFromImage(itk.imread(img_path))

    def test_ants():
        for img_path in img_paths:
            array = ants.image_read(img_path).numpy()

    nib_start = time.time()
    for i in range(N_TRIALS):
        test_nibabel()
    nib_end = time.time()
    print('NIBABEL TIME: %.3f seconds' % (nib_end-nib_start))

    itk_start = time.time()
    for i in range(N_TRIALS):
        test_itk()
    itk_end = time.time()
    print('ITK TIME: %.3f seconds' % (itk_end-itk_start))

    ants_start = time.time()
    for i in range(N_TRIALS):
        test_ants()
    ants_end = time.time()
    print('ANTS TIME: %.3f seconds' % (ants_end-ants_start))


if __name__ == '__main__':
    time_nifti_to_numpy(N_TRIALS=1)
    time_nifti_to_numpy(N_TRIALS=20)

