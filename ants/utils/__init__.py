from .channels import merge_channels, split_channels
from .consistency import image_physical_space_consistency, allclose
from .get_ants_data import get_ants_data, get_data
from .matrix_image import (matrix_to_images,
                           images_from_matrix,
                           image_list_to_matrix,
                           images_to_matrix,
                           matrix_from_images,
                           timeseries_to_matrix,
                           matrix_to_timeseries)
from .mni2tal import mni2tal
from .ndimage_to_list import ndimage_to_list, list_to_ndimage
from .nifti_to_ants import nifti_to_ants
from .scalar_rgb_vector import rgb_to_vector, vector_to_rgb, scalar_to_rgb