"""
Get local ANTsPy data
"""

__all__ = ['get_ants_data']

import os 
data_path = os.path.expanduser('~/.antspy/')

def get_ants_data(name):
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
            - 'mni'
            - 'surf'
    Returns
    -------
    string
        filepath of test image
    """
    if name == 'r16':
        datapath = os.path.join(data_path, 'r16slice.jpg')
    elif name == 'r27':
        datapath = os.path.join(data_path, 'r27slice.jpg')
    elif name == 'r64':
        datapath = os.path.join(data_path, 'r64slice.jpg')
    elif name == 'r85':
        datapath = os.path.join(data_path, 'r85slice.jpg')
    elif name == 'mni':
        datapath = os.path.join(data_path, 'mni.nii.gz')
    elif name == 'surf':
        datapath = os.path.join(data_path, 'surf.nii.gz')
    else:
        raise ValueError('data file not found')

    return datapath