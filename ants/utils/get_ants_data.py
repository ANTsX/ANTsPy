"""
Get local ANTsPy data
"""

__all__ = ['get_ants_data',
           'get_data']

import os 
data_path = os.path.expanduser('~/.antspy/')

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
    if name is None:
        files = []
        for fname in os.listdir(data_path):
            if (fname.endswith('.nii.gz')) or (fname.endswith('.jpg') or (fname.endswith('.csv'))):
                fname = os.path.join(data_path, fname)
                files.append(fname)
        return files
    else:
        datapath = None
        for fname in os.listdir(data_path):
            if (name == fname.split('.')[0]) or ((name+'slice') == fname.split('.')[0]):
                datapath = os.path.join(data_path, fname)

        if datapath is None:
            raise ValueError('File doesnt exist. Options: ' , os.listdir(data_path))
        return datapath

get_ants_data = get_data




