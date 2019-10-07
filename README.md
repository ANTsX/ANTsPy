

# Advanced Normalization Tools in Python

![img](https://media0.giphy.com/media/OCMGLUo7d5jJ6/200_s.gif)
<br>

[![CircleCI](https://circleci.com/gh/ANTsX/ANTsPy.svg?style=svg)](https://circleci.com/gh/ANTsX/ANTsPy)
[![Build Status](https://travis-ci.org/ANTsX/ANTsPy.svg?branch=master)](https://travis-ci.org/ANTsX/ANTsPy)
<a href='https://coveralls.io/github/ANTsX/ANTsPy?branch=master'><img src='https://coveralls.io/repos/github/ANTsX/ANTsPy/badge.svg?branch=master' alt='Coverage Status' /></a>
<a href='http://antspyx.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/antspyx/badge/?version=latest' alt='Documentation Status' />
</a>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/stnava/ANTsPyDocker/master)

## About ANTsPy

Search [ANTsPy documentation at read the docs.](https://antspyx.readthedocs.io/en/latest/?badge=latest)

<i>ANTsPy</i> is a Python library which wraps the C++ biomedical image processing library <i>[ANTs](https://github.com/ANTsX/ANTs)</i>,
matches much of the statistical capabilities of <i>[ANTsR](https://github.com/ANTsX/ANTsR)</i>, and allows seamless integration
with <i>numpy</i>, <i>scikit-learn</i>, and the greater Python community.

<i>ANTsPy</i> includes blazing-fast IO (~40% faster than <i>nibabel</i> for loading Nifti images and
converting them to <i>numpy</i> arrays), registration, segmentation, statistical learning,
visualization, and other useful utility functions.

<i>ANTsPy</i> also provides a low-barrier opportunity for users to quickly wrap their <i>ITK</i> (or general C++)
code in Python without having to build an entire IO/plotting/wrapping code base from
scratch - see [C++ Wrap Guide](tutorials/UsingITK.md) for a succinct tutorial.

If you want to contribute to <i>ANTsPy</i> or simply want to learn about the package architecture
and wrapping process, please read the extensive [contributors guide](CONTRIBUTING.md).

If you have any questions or feature requests, feel free to open an issue or email Nick (ncullen at pennmedicine dot upenn dot edu).

## Installation

We recommend that users install the latest pre-compiled binaries, which takes ~1 minute. Note
that <i>ANTsPy</i> is not currently tested for Python 2.7 support.
Copy the following command and paste it into your bash terminal:

For MacOS:
```bash
pip install https://github.com/ANTsX/ANTsPy/releases/download/Weekly/antspy-0.1.4-cp36-cp36m-macosx_10_7_x86_64.whl
```

For Linux:
```bash
pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
```

If the above doesn't work on your platform, then use:

```
git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
python3 setup.py install
```
if you want more detailed instructions on installing <i>ANTsPy</i>, you can
read the [installation tutorial](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/InstallingANTsPy.md).

------------------------------------------------------------------------------
## ITK & VTK

#### Insight Toolkit (ITK)

By default, <i>ANTsPy</i> will search for an existing ITK build by checking if the `ITK_DIR`
environment variable is set. If that is not
found, it will build it for you. It does <b>NOT</b> require the Python wrappings for
ITK.

#### Visualization Toolkit (VTK)

By default, <i>ANTsPy</i> will search for an existing VTK build by checking if the `VTK_DIR`
environment variable is set. If that is not
found, it will build it for you. It does <b>NOT</b> require the Python wrappings for
VTK. If you do not want VTK, then add the `--novtk` flag to setup (e.g. `python setup.py install --novtk`).

## ANTsR Comparison

Here is a quick example to show the similarity with <i>ANTsR</i>:

<i>ANTsR</i> code:
```R
library(ANTsR)
img   <- antsImageRead(getANTsRData("r16"))
img   <- resampleImage(img, c(64,64), 1, 0 )
mask  <- getMask(img)
segs1 <- atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask )
```

<i>ANTsPy</i> code:
```python     
import ants
img   = ants.image_read(ants.get_data("r16"))
img   = ants.resample_image(img, (64,64), 1, 0 )
mask  = ants.get_mask(img)
segs1 = ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask )
```

## Tutorials

We provide numerous tutorials for new users: [https://github.com/ANTsX/ANTsPy/tree/master/tutorials](https://github.com/ANTsX/ANTsPy/tree/master/tutorials)

[5 minute Overview](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/tutorial_5min.md)

[Nibabel Speed Comparison](https://github.com/ANTsX/ANTsPy/blob/master/tests/timings_io.py)

[Composite registrations](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistrations.ipynb)

## Build documentation

```
cd docs
sphinx-apidoc -o source/ ../
make html
```
