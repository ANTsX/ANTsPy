"""
BIDSCohort class for handling BIDS datasets.
This class allows you to streamline various tasks
related to neuroimage processing, machine learning 
analysis, and deep learning sampling


NOTES
------
# Example: run N4 bias correction on the T1 images of a BIDS dataset
cohort = BIDSCohort(directory='~/desktop/projects/dlbs-bet/data/raw-bids/')

def n4(img):
    return img.n4_bias_correction()

cohort.apply_to_images(fn=n4, modality='T1w', subjects='*', 
                       out_suffix='N4')



"""

import ants

from bids.grabbids import BIDSLayout


class Cohort(object):
    """
    Base class for Cohort objects. This class allows
    you to streamline the processing and analysis of
    medical imaging datasets
    """
    def __init__(self):
        pass


class CSVCohort(Cohort): pass
class FolderCohort(Cohort): pass
class ListCohort(Cohort): pass


class CSVSampler(object):
    def __init__(self, dataframe, input_reader=None, target_reader=None, 
        input_transform=None, target_transform=None, co_transform=None,
        input_return_processor=None, target_return_processor=None, co_return_processor=None):
        pass

    def generate(self):
        """
        Return a generator that can be passed in to Keras `fit_generator`
        """
        pass

class BIDSCohort(BIDSLayout):
    """
    BIDSCohort class for handling BIDS datasets.
    """
    def __init__(self, path, **kwargs):
        """
        Initialize a BIDS cohort object
        """
        super(BIDSCohort, self).__init__(path=path, **kwargs)

    def apply_to_images(self, fn, modality, image_type=None, subjects='*', out_suffix=''):
        for subject in self.subjects:
            in_file = self.get_modality(subject=subject, modality=modality)

            img = ants.image_read(in_file)
            img_proc = fn(img)

            out_file = in_file.replace('.nii.gz', '%s.nii.gz' % out_suffix)
            ants.image_write(img_proc, out_file)

    def __getitem__(self, index):
        """
        Access items from the cohort.

        Arguments
        ---------
        index : string
            if index is a subject ID, then a dictionary will be returned
            where keys are the available modalities and values is the 
            file path or list of file paths available for that modality
        """
        pass

    def create_sampler(self, inputs, targets, input_reader=None, target_reader=None, 
        input_transform=None, target_transform=None, co_transform=None,
        input_return_processor=None, target_return_processor=None, co_return_processor=None):
        """
        Create a BIDSSampler that can be used to generate infinite augmented samples
        """
        pass


    def copy_structure(self, other_directory):
        """
        Copy the folder structure of the BIDSCohort
        to another base directory, without any of the
        actual files being copied. 

        This is useful for creating a separate BIDSCohort 
        for processed data.
        """
        pass



