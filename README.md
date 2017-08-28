
# Advanced Normalization Tools in Python

Welcome to the Derek Zoolander Center for Kids who cant analyze brain images, and 
wanna learn how to do other stuff good, too. <br>
<br>
What is this, a center for ANTs? <br>
How can we expect to teach children to analyze brain images, if they can't
even fit in the building? <br>
It needs to be at least... three times bigger than this. <br>

## What is ANTsPy?

ANTsPy -- pronounced "ant-spy" like an ant who is a secret agent -- is a Python library which
wraps much of the medical image processing functionality of ANTs, 
provides much of the statistical capabilities of ANTsR, and allows seemless integration
with Numpy and the greater Python community. 

The engine underlying AntsPy and ANTsR is the same, so you can generally expect exactly the same results
between the two packages <b>given that the algorithm in question is itself deterministic</b>.

## Installation

To install, run the following:
```bash
git clone https://github.com/ANTsConsortium/ANTsPy.git
cd ANTsPy
python setup.py develop
```

By default, ANTsPy will search for an existing ITK installation. If that is not
found, it will install it for you. If you want to use 3D visualization tools
such as `ants.Surf` or `ants.Vol`, you need to install VTK on your own right now.

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
    * e.g. antsrSurf -> ants.Surf

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

### Chaining Commands
In ANTsR you can use the `%>%` command to chain operations. That's real nice. In ANTsPy, you can 
do this automatically on ANTsImages. Amazing stuff..

```python
import ants
img = ants.image_read(ants.get_ants_data('r16'))
img = img.resample_image((64,64), 1, 0).get_mask().atropos(m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
```

