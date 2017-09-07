
# Advanced Normalization Tools in Python
![img](https://media0.giphy.com/media/OCMGLUo7d5jJ6/200_s.gif) <br>

## What is ANTsPy?

ANTsPy -- pronounced "ant-spy" like an ant who is a secret agent -- is a Python library which
wraps much of the medical image processing functionality of ANTs, 
provides much of the statistical capabilities of ANTsR, and allows seamless integration
with Numpy and the greater Python community. 

The engine underlying AntsPy and ANTsR is the same, so you can generally expect exactly the same results
between the two packages <b>given that the algorithm in question is itself deterministic</b>.

## Installation

```bash
git clone https://github.com/ANTsConsortium/ANTsPy.git
cd ANTsPy
python setup.py install
```

#### Insight Toolkit (ITK)

By default, ANTsPy will search for an existing ITK build by checking if the `ITK_DIR`
environment variable is set. If that is not
found, it will build it for you. It does <b>NOT</b> require the Python wrappings for
ITK. If you want to use 3D visualization tools
such as `ants.Surf` or `ants.Vol`, you need to build VTK on your own right now.

ANTsPy is known to install on MacOS and Ubuntu, both with Python3.6. It's unlikely that
it will work with Python2.7.

## ANTsR Comparison

Here are a few example to get you up-and-running if coming from ANTsR:

### Example 1

ANTsR code:
```R
library(ANTsR)
img <- antsImageRead( getANTsRData("r16") , 2 )
img <- resampleImage( img, c(64,64), 1, 0 )
mask <- getMask(img)
segs1 <- atropos( a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
```

ANTsPy code:
```python     
import ants
img = ants.image_read(ants.get_ants_data('r16'))
img = ants.resample_image(img, (64,64), 1, 0)
mask = ants.get_mask(img)
seg1 = ants.atropos(a = img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
```


## Quick Tutorial

ANTsPy functions and classes are generally faithful to the respective ANTsR versions,
with the following consistent changes in naming convention:<br>
* camel case in ANTsR is underscore case in ANTsPy
    * e.g. resampleImage -> resample_image
* anything preceeded by `ants` or `antsr` in ANTsR is removed since ANTsPy uses namespaces already
    * e.g. antsImageRead -> ants.image_read

### Read an Image

```python
import ants
img = ants.image_read( ants.get_ants_data('r16') )
print(img)
```

### Image Properties

Image properties are pythonic and easy to get/set:

```python
import ants
img = ants.image_read( ants.get_ants_data('r16') )

print(img.spacing)
img.set_spacing( (2., 2.) )

print(img.origin)
img.set_origin( (100,100) )
```

However, we still try to keep most of the associated ANTsR functions which are stand-alone, e.g:

```python
ants.get_spacing(img) # versus the pythonic `img.spacing`
ants.set_spacing(img, (2.,2.)) # versus `img.set_spacing`

imgclone = ants.image_clone(img) # versus img.clone()
```

### Converting to Numpy

ANTsPy provides seamless conversions to Numpy arrays. Through the use
of memory buffers directly in the C++ api, these calls are instantaneous and essentially free.

```python
import ants
img = ants.image_read( ants.get_ants_data('mni') )
img_array = img.numpy()
```

Do operations directly in numpy if you want, then simply make an ANTsImage right back
from the numpy array (again instantaneous and "free"):

```python
import ants
img = ants.image_read( ants.get_ants_data('mni') )
img_array = img.numpy()

img_array += 5

# copies image information and just changes the data
new_img1 = img.new_image_like(img_array)

# doesnt copy any information
new_img2 = ants.from_numpy(img_array)

# verbose way to copy information
new_img3 = ants.from_numpy(img_array, spacing=img.spacing,
                           origin=img.origin, direction=img.direction)
```

### Indexing 

Images can be indexed (getting and setting) exactly as if they were arrays.
```python
import ants
img = ants.image_read( ants.get_ants_data('mni') )

vals = img[200,:,:] # get a slice

img[100,:,:] = 1 # set a slice
```

### Operator Overloading

All common mathematical operators are overloaded to work directly on ANTsImages:

```python
import ants
import numpy as np
img = ants.image_read( ants.get_ants_data('mni') )
img2 = img.clone()
img3 = img + img2
print(np.allclose(img.numpy()+img2.numpy(), img3.numpy())) # same as if done in numpy
```

### Chaining Commands
In ANTsR you can use the `%>%` command to chain operations. That's real nice. In ANTsPy, you can 
do this automatically on ANTsImages. Amazing stuff..

```python
import ants
img = ants.image_read(ants.get_ants_data('r16'))
img = img.resample_image((64,64), 1, 0).get_mask().atropos(m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
```

## Frequently Asked Questions
* How long does ANTsPy take to install?
    * 15-20 minutes if you don't have a local ITK build


