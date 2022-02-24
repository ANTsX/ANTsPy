"""
Get local ANTsPy data
"""

__all__ = ['get_ants_data',
           'get_data']

import os
import requests
import tempfile

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
            "surf": "https://ndownloader.figshare.com/files/28726530"
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
                  "surf"
                  "show")

    if file_id == "show" or file_id is None:
       return(valid_list)

    url = switch_data(file_id)

    if target_file_name == None:
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
