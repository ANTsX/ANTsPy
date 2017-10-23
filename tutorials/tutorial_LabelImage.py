"""
# A tutorial about Label Images in ANTsPy

In ANTsPy, we have a special class for dealing with what I call 
"Label Images" - a brain image where each pixel/voxel is associated with 
a specific label. For instance, an atlas or parcellation is the prime example 
of a label image. But `LabelImage` types dont <i>just</i> have labels... they 
also can have real values associated with those labels. For instance, suppose 
you have a set of Cortical Thickness values derived from an atlas, and you want 
to assign those regional values *back* onto an actual brain image for plotting 
or to perform analysis tasks which require some notion of spatial location. 
`LabelImage` types let you do this.

Basically, to create a label image in *ANTsPy*, you need two things (one is 
optional but highly recommended):
- a discrete atlas image (a normal `ANTsImage` type)
- (optionally) a pandas dataframe or python dictionary with a mapping 
  from discrete values in the atlas image to string atlas labels

This tutorial will show you all the beautiful things you can do with `LabelImage` types.
"""


"""
## A simple example

We will start with a simple example to demonstrate label images - a 2D square
with four regions
"""

import ants
import os
import numpy as np
import pandas as pd


# create discrete image
square = np.zeros((20,20))
square[:10,:10] = 0
square[:10,10:] = 1
square[10:,:10] = 2
square[10:,10:] = 3

# create regular ANTsImage from numpy array
img = ants.from_numpy(square).astype('uint8')

# plot image
#img.plot(cmap=None)

"""
Above, we created our discrete "atlas" image. Next, we will
create a dictionary containing the names for each value in 
the atlas. We will make simple names.
"""

label_df = np.asarray([['TopRight',    'Right', 'Top'],
                       ['BottomRight', 'Right', 'Bottom'],
                       ['TopLeft',     'Left',  'Top'],
                       ['BottomLeft',  'Left',  'Bottom']])

label_df = pd.DataFrame(label_df, index=[1,2,3,4],
                        columns=['Quadrant', 'Right/Left', 'Top/Bottom'])

atlas = ants.LabelImage(label_image=img, label_info=label_df)


"""
You can index a label image like a dictionary, and it will return
the unique image values corresponding to that label, or more than
one if appropriate.
"""
up_right_idx = atlas['UpperRight']
print(up_right_idx) # should be 1

right_idxs = atlas['Right']
print(right_idxs) # should be [1, 2]


"""
## A real example

Now that we have the basics of the `ants.LabelImage` class down, we 
can move on to a real example to show how this would work in practice.

In this example, we have a Freesurfer atlas (the Desikan-killany atlas,
aka "aparc+aseg.mgz") and a data frame of aggregated cortical thickness values 
for a subset of those regions for a collection of subjects.

Our first task is to create a LabelImage for this atlas.
"""

"""
We start by loading in the label info as a pandas dataframe
"""

proc_dir = '/users/ncullen/desktop/projects/tadpole/data/processed/'
raw_dir  = '/users/ncullen/desktop/projects/tadpole/data/raw/'

label_df = pd.read_csv(os.path.join(proc_dir, 'UCSF_FS_Map.csv'), index_col=0)

print(label_df.head())

"""
As you can see, the label dataframe has the the atlas values as the dataframe
index and a set of columns with different labels for each index.

Next, we load in the discrete atlas image.
"""

atlas_img = ants.image_read(os.path.join(raw_dir, 'freesurfer/aparc+aseg.mgz')).astype('uint32')
atlas_img.plot()

label_img = ants.LabelImage(image=atlas_img, info=label_df)

"""
Let's see this in action on a template
"""
t1_img = ants.image_read(os.path.join(raw_dir,'freesurfer/T1.mgz'))
t1_img.plot()

# set the label image
t1_img.set_label_image(atlas_img)




"""
Our second task is create an image for each subject that fills in the brain
region locations with the associated region's cortical thickness
"""
data = pd.read_csv(os.path.join())























