"""
Get local ANTsPy data
"""

__all__ = ['get_ants_data',
           'get_data']

import os

DATA_PATH = os.path.expanduser('~/.antspy/')

def get_data(name=None):
    """
    Get ANTsPy test data filename

    ANTsR function: `getANTsRData`

    Arguments
    ---------
    name : string
        name of test image tag to retrieve
        Options:
            - 'r16'
            - 'r27'
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
    os.makedirs(DATA_PATH, exist_ok=True)

    if name is None:
        files = []
        for fname in os.listdir(DATA_PATH):
            if (fname.endswith('.nii.gz')) or (fname.endswith('.jpg') or (fname.endswith('.csv'))):
                fname = os.path.join(DATA_PATH, fname)
                files.append(fname)
        return files

    datapath = None
    for fname in os.listdir(DATA_PATH):
        if (name == fname.split('.')[0]) or ((name+'slice') == fname.split('.')[0]):
            datapath = os.path.join(DATA_PATH, fname)

    if datapath is None:
        raise ValueError('File doesnt exist. Options: ' , os.listdir(DATA_PATH))
    return datapath

get_ants_data = get_data
