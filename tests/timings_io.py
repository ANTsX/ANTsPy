"""
Timings against numpy/itk/nibabel/ants

To run the timings, first install itk, nibabel, and ants.
Then, run this file `timings_io.py` (e.g. `python timings_io.py`)

On a Macbook Pro
---------------
1 Trial:
NIBABEL TIME: 2.902 seconds
ITK TIME: 4.375 seconds
ANTS TIME: 1.713 seconds

20 Trials:
NIBABEL TIME: 56.121 seconds
ITK TIME: 28.780 seconds
ANTS TIME: 33.601 seconds
"""

import os

import numpy as np
import nibabel as nib
import itk
import ants

import time

def time_nifti_to_numpy(N_TRIALS):
    """
    Times how fast a framework can read a nifti file and convert it to numpy
    """
    img_paths = [ants.get_ants_data('mni')]*10
    
    def test_nibabel():
        for img_path in img_paths:
            array = np.asanyarray(nib.load(img_path).dataobj)

    def test_itk():
        for img_path in img_paths:
            array = itk.GetArrayFromImage(itk.imread(img_path))

    def test_ants():
        for img_path in img_paths:
            array = ants.image_read(img_path, pixeltype='float').numpy()

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
    time_nifti_to_numpy(N_TRIALS=1) # 
    time_nifti_to_numpy(N_TRIALS=20)

