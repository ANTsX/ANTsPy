import os
import requests
import tempfile

from . import lib
from . import ants_image as iio
import numpy as np


def copy_image_info(reference, target):
    """
    Copy origin, direction, and spacing from one antsImage to another

    ANTsR function: `antsCopyImageInfo`

    Arguments
    ---------
    reference : ANTsImage
        Image to get values from.
    target  : ANTsImAGE
        Image to copy values to

    Returns
    -------
    ANTsImage
        Target image with reference header information
    """
    target.set_origin(reference.origin)
    target.set_direction(reference.direction)
    target.set_spacing(reference.spacing)
    return target

def image_physical_space_consistency(image1, image2, tolerance=1e-2, datatype=False):
    """
    Check if two or more ANTsImage objects occupy the same physical space

    ANTsR function: `antsImagePhysicalSpaceConsistency`

    Arguments
    ---------
    *images : ANTsImages
        images to compare

    tolerance : float
        tolerance when checking origin and spacing

    data_type : boolean
        If true, also check that the image data types are the same

    Returns
    -------
    boolean
        true if images share same physical space, false otherwise
    """
    images = [image1, image2]

    img1 = images[0]
    for img2 in images[1:]:
        if (not isinstance(img1, ANTsImage)) or (not isinstance(img2, ANTsImage)):
            raise ValueError('Both images must be of class `AntsImage`')

        # image dimension check
        if img1.dimension != img2.dimension:
            return False

        # image spacing check
        space_diffs = sum([abs(s1-s2)>tolerance for s1, s2 in zip(img1.spacing, img2.spacing)])
        if space_diffs > 0:
            return False

        # image origin check
        origin_diffs = sum([abs(s1-s2)>tolerance for s1, s2 in zip(img1.origin, img2.origin)])
        if origin_diffs > 0:
            return False

        # image direction check
        origin_diff = np.allclose(img1.direction, img2.direction, atol=tolerance)
        if not origin_diff:
            return False

        # data type
        if datatype == True:
            if img1.pixeltype != img2.pixeltype:
                return False

            if img1.components != img2.components:
                return False

    return True


def image_type_cast(image_list, pixeltype=None):
    """
    Cast a list of images to the highest pixeltype present in the list
    or all to a specified type

    ANTsR function: `antsImageTypeCast`

    Arguments
    ---------
    image_list : list/tuple
        images to cast

    pixeltype : string (optional)
        pixeltype to cast to. If None, images will be cast to the highest
        precision pixeltype found in image_list

    Returns
    -------
    list of ANTsImages
        given images casted to new type
    """
    if not isinstance(image_list, (list,tuple)):
        raise ValueError('image_list must be list of ANTsImage types')

    pixtypes = []
    for img in image_list:
        pixtypes.append(img.pixeltype)

    if pixeltype is None:
        pixeltype = 'unsigned char'
        for p in pixtypes:
            if p == 'double':
                pixeltype = 'double'
            elif (p=='float') and (pixeltype!='double'):
                pixeltype = 'float'
            elif (p=='unsigned int') and (pixeltype!='float') and (pixeltype!='double'):
                pixeltype = 'unsigned int'

    out_images = []
    for img in image_list:
        if img.pixeltype == pixeltype:
            out_images.append(img)
        else:
            out_images.append(img.clone(pixeltype))

    return out_images


def allclose(image1, image2):
    """
    Check if two images have the same array values
    """
    return np.allclose(image1.numpy(), image2.numpy())


def get_data(file_id=None, target_file_name=None, antsx_cache_directory=None):
    """
    Get ANTsPy test data file

    ANTsR function: `getANTsRData`

    Arguments
    ---------
    name : string
        name of test image tag to retrieve
        Options:
            - 'r16'
            - 'r27'
            - 'r30'
            - 'r62'
            - 'r64'
            - 'r85'
            - 'ch2'
            - 'mni'
            - 'surf'
            - 'pcasl'
    Returns
    -------
    string
        filepath of test image

    Example
    -------
    >>> import ants
    >>> mnipath = ants.get_ants_data('mni')
    """

    def switch_data(argument):
        switcher = {
            "r16": "https://ndownloader.figshare.com/files/28726512",
            "r27": "https://ndownloader.figshare.com/files/28726515",
            "r30": "https://ndownloader.figshare.com/files/28726518",
            "r62": "https://ndownloader.figshare.com/files/28726521",
            "r64": "https://ndownloader.figshare.com/files/28726524",
            "r85": "https://ndownloader.figshare.com/files/28726527",
            "ch2": "https://ndownloader.figshare.com/files/28726494",
            "mni": "https://ndownloader.figshare.com/files/28726500",
            "surf": "https://ndownloader.figshare.com/files/28726530",
            "pcasl": "http://files.figshare.com/1862041/101_pcasl.nii.gz",
        }
        return(switcher.get(argument, "Invalid argument."))

    if antsx_cache_directory is None:
        antsx_cache_directory = os.path.expanduser('~/.antspy/')
    os.makedirs(antsx_cache_directory, exist_ok=True)

    if os.path.isdir(antsx_cache_directory) == False:
        antsx_cache_directory = tempfile.TemporaryDirectory()

    valid_list = ("r16",
                  "r27",
                  "r30",
                  "r62",
                  "r64",
                  "r85",
                  "ch2",
                  "mni",
                  "surf",
                  "pcasl",
                  "show")

    if file_id == "show" or file_id is None:
       return(valid_list)

    url = switch_data(file_id)

    if target_file_name == None:
        if file_id == "pcasl":
            target_file_name = antsx_cache_directory + "pcasl.nii.gz"
        else:
            extension = ".jpg"
            if file_id == "ch2" or file_id == "mni" or file_id == "surf":
                extension = ".nii.gz"
            if extension == ".jpg":
                target_file_name = antsx_cache_directory + file_id + "slice" + extension
            else:
                target_file_name = antsx_cache_directory + file_id + extension

    target_file_name_path = target_file_name
    if target_file_name == None:
        target_file = tempfile.NamedTemporaryFile(prefix=target_file_name, dir=antsx_cache_directory)
        target_file_name_path = target_file.name
        target_file.close()

    if not os.path.exists(target_file_name_path):
        r = requests.get(url)
        with open(target_file_name_path, 'wb') as f:
            f.write(r.content)

    return(target_file_name_path)

get_ants_data = get_data
