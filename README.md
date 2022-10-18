

# Advanced Normalization Tools in Python

![img](https://media0.giphy.com/media/OCMGLUo7d5jJ6/200_s.gif)
<br>

[![Build Status](https://travis-ci.org/ANTsX/ANTsPy.svg?branch=master)](https://travis-ci.org/ANTsX/ANTsPy)
<a href='https://coveralls.io/github/ANTsX/ANTsPy?branch=master'><img src='https://coveralls.io/repos/github/ANTsX/ANTsPy/badge.svg?branch=master' alt='Coverage Status' /></a>
<a href='http://antspyx.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/antspyx/badge/?version=latest' alt='Documentation Status' />
</a>
[![ci-docker](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)
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

For MacOS and Linux:
```bash
pip install antspyx
```

If we do not have releases for your platform, then use:

```
git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
python3 setup.py install
```
if you want more detailed instructions on compiling <i>ANTsPy</i> from source, you can
read the [installation tutorial](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/InstallingANTsPy.md).

NOTE: we are hoping to relatively soon release windows wheels via `pip`.
If they are not yet available, please [check the discussion in the issues](https://github.com/ANTsX/ANTsPy/issues/301)
for how to build from source on windows machines.

### Recent wheels

Look under the "Actions" tab.  Then click on the commit for the software version you want.
Wheels for some of these commits will be available by downloading its "artifacts".

### Docker images

Available on [Docker Hub](https://hub.docker.com/repository/docker/antsx/antspy). To build
ANTsPy docker images, see the (installation tutorial)(https://github.com/ANTsX/ANTsPy/blob/master/tutorials/InstallingANTsPy.md#docker-installation).

------------------------------------------------------------------------------

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
from ants import atropos, get_ants_data, image_read, resample_image, get_mask
img   = image_read(get_ants_data("r16"))
img   = resample_image(img, (64,64), 1, 0 )
mask  = get_mask(img)
segs1 = atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask )
```

## Tutorials

We provide numerous tutorials for new users: [https://github.com/ANTsX/ANTsPy/tree/master/tutorials](https://github.com/ANTsX/ANTsPy/tree/master/tutorials)

[5 minute Overview](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/tutorial_5min.md)

[Nibabel Speed Comparison](https://github.com/ANTsX/ANTsPy/blob/master/tests/timings_io.py)

[Composite registrations](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistrations.ipynb)

## other notes on compilation

in some cases, you may need some other libraries if they are not already installed eg if cmake says something about
a missing png library or a missing `Python.h` file.

```
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install gfortran
sudo apt-get install libpng-dev
sudo apt-get install python3-dev  # for python3.x installs
```

## Build documentation

```
cd docs
sphinx-apidoc -o source/ ../
make html
```

## References

1. See references at the main [ANTs page](https://github.com/ANTsX/ANTs#boilerplate-ants).

2. [Google scholar search reveals plenty of explanation of methods and evaluation results by ourselves](https://scholar.google.com/scholar?start=0&q=advanced+normalization+tools+ants+image+registration&hl=en&as_sdt=0,40)

3. [ANTs evaluation and comparison by other authors](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C40&q=advanced+normalization+tools+ants+image+registration+-avants+-tustison&btnG=)
