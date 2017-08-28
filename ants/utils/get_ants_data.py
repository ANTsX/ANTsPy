"""
Get local ANTsPy data
"""

__all__ = ['get_ants_data']

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join('/'.join(dir_path.split('/')[:-2]), 'data')

def get_ants_data(name): 
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