# Advanced Normalization Tools in Python

Round 3

[![Coverage Status](https://coveralls.io/repos/github/ANTsX/ANTsPy/badge.svg?branch=master)](https://coveralls.io/github/ANTsX/ANTsPy?branch=master)
<a href='http://antspyx.readthedocs.io/en/latest/?badge=latest'>
</a>
![PyPI - Downloads](https://img.shields.io/pypi/dm/antspyx?label=pypi%20downloads)
[![Build](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels.yml)
[![ci-docker](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/antsx/antspy.svg)](https://hub.docker.com/repository/docker/antsx/antspy)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)
[![PubMed](https://img.shields.io/badge/ANTsX_paper-Open_Access-8DABFF?logo=pubmed)](https://pubmed.ncbi.nlm.nih.gov/33907199/)

<br>

ANTsPy is a Python library which wraps the well-established C++ biomedical image processing library <i>[ANTs](https://github.com/ANTsX/ANTs)</i>. It includes blazing-fast reading and writing of medical images, algorithms for registration, segmentation, and statistical learning, as well as functions to create publication-ready visualizations.

If you are looking to train deep learning models on your medical images, you might be interested in [ANTsPyNet](https://github.com/ANTsX/ANTsPy) which provides tools for training and visualizing deep learning models. ANTsPy and ANTsPyNet seamlessly integrate with the greater Python community, particularly deep learning libraries, scikit-learn, and numpy.

<br>

## Installation

We recommend that users install the latest pre-compiled binaries, which takes ~1 minute. 

For MacOS and Linux:

```bash
pip install antspyx
```

Because of limited storage space, pip binaries are not available for every combination of python
version and platform. Additionally, we are required to remove outdated wheels. If we do not have releases for your platform, you can check the
[Github Releases page](https://github.com/ANTsX/ANTsPy/releases) or build from source with:

```
git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
python3 setup.py install
```

If you want more detailed instructions
on compiling <i>ANTsPy</i> from source, you can read the
[installation tutorial](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/InstallingANTsPy.md).

### Installing specific versions

We cannot store the entire history of releases because storage space on `pip` is limited. If you need an older release, you can check the [Github Releases page](https://github.com/ANTsX/ANTsPy/releases) or
build from source. To install a specific version from source, you can try the following:

```bash
pip install 'antspyx @ git+https://github.com/ANTsX/ANTsPy@v0.3.2'
```

which will attempt to build from source (requires a machine with developer tools).

### Recent wheels

Non-release commits have wheels built automatically, which are available for download for a limited period.
Look under the [Actions tab](https://github.com/ANTsX/ANTsPy/actions). Then click on the commit for the software version you want.
Recent commits will have wheels stored as "artifacts".

Wheels are built locally like this:

```
rm -r -f build/ antspymm.egg-info/ dist/
python3 setup.py sdist bdist_wheel
pipx run twine upload dist/*
```

### Docker images

Available on [Docker Hub](https://hub.docker.com/repository/docker/antsx/antspy). To build
ANTsPy docker images, see the (installation tutorial)(https://github.com/ANTsX/ANTsPy/blob/master/tutorials/InstallingANTsPy.md#docker-installation).

---

<br>

## Quickstart

Here is an example of reading in an image, using various utility functions such as resampling and masking, then performing three-class Atropos segmentation.

```python
from ants import atropos, get_ants_data, image_read, resample_image, get_mask
img   = image_read(get_ants_data("r16"))
img   = resample_image(img, (64,64), 1, 0 )
mask  = get_mask(img)
segs1 = atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask )
```

<br>

## Tutorials

We provide numerous tutorials for new users: [https://github.com/ANTsX/ANTsPy/tree/master/tutorials](https://github.com/ANTsX/ANTsPy/tree/master/tutorials)

[5 minute Overview](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/tutorial_5min.md)

[Nibabel Speed Comparison](https://github.com/ANTsX/ANTsPy/blob/master/tests/timings_io.py)

[Composite registrations](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistrations.ipynb)

<br>

## Other notes on compilation

In some cases, you may need some other libraries if they are not already installed eg if cmake says something about
a missing png library or a missing `Python.h` file.

```
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install gfortran
sudo apt-get install libpng-dev
sudo apt-get install python3-dev  # for python3.x installs
```

### Build documentation

```
cd docs
sphinx-apidoc -o source/ ../
make html
```

<br>

## References

1. See references at the main [ANTs page](https://github.com/ANTsX/ANTs#boilerplate-ants).

2. [Google scholar search reveals plenty of explanation of methods and evaluation results by ourselves](https://scholar.google.com/scholar?start=0&q=advanced+normalization+tools+ants+image+registration&hl=en&as_sdt=0,40)

3. [ANTs evaluation and comparison by other authors](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C40&q=advanced+normalization+tools+ants+image+registration+-avants+-tustison&btnG=)
