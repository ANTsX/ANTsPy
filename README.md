# Advanced Normalization Tools in Python

[![Coverage Status](https://coveralls.io/repos/github/ANTsX/ANTsPy/badge.svg?branch=master)](https://coveralls.io/github/ANTsX/ANTsPy?branch=master)
<a href='http://antspyx.readthedocs.io/en/latest/?badge=latest'>
</a>
![PyPI - Downloads](https://img.shields.io/pypi/dm/antspyx?label=pypi%20downloads)
[![Nightly Build](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels.yml)
[![ci-pytest](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-pytest.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-pytest.yml)
[![ci-docker](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/antsx/antspy.svg)](https://hub.docker.com/repository/docker/antsx/antspy)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)
[![PubMed](https://img.shields.io/badge/ANTsX_paper-Open_Access-8DABFF?logo=pubmed)](https://pubmed.ncbi.nlm.nih.gov/33907199/)

<br>

The ANTsPy library wraps the well-established C++ biomedical image processing framework <i>[ANTs](https://github.com/antsx/ants)</i>. It includes blazing-fast reading and writing of medical images, algorithms for registration, segmentation, and statistical learning, as well as functions to create publication-ready visualizations.

If you are looking to train deep learning models on your medical images, you might be interested in [antspynet](https://github.com/antsx/antspy) which provides tools for training and visualizing deep learning models. ANTsPy and ANTsPyNet seamlessly integrate with the greater Python community, particularly deep learning libraries, scikit-learn, and numpy.

<br>

## Installation

The easiest way to install ANTsPy is via the latest pre-compiled binaries from PyPI.

```bash
pip install antspyx
```

Because of limited storage space, pip binaries are not available for every combination of python
version and platform. If we do not have releases for your platform, you can check the
[Github Releases page](https://github.com/antsx/antspy/releases) or build from source:

```
git clone https://github.com/antsx/antspy
cd antspy
python -m pip install .
```

Further details about installing ANTsPy or building it from source can be found in the
[installation tutorial](https://github.com/antsx/antspy/blob/master/tutorials/Installation.md).

<br>

## Quickstart

Here is an example of reading in an image, using various utility functions such as resampling and masking, then performing three-class Atropos segmentation.

```python
import ants
img   = ants.image_read(get_data("r16"))
img   = ants.resample_image(img, (64,64), 1, 0 )
mask  = ants.get_mask(img)
segs1 = ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask)
```

<br>

## Tutorials

Resources for learning about ANTsPy can be found in the [tutorials](https://github.com/ANTsX/ANTsPy/tree/master/tutorials) folder. An overview of the available tutorials is presented below.

- [Basic overview](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/tutorial_5min.md)

- [Composite registrations](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistrations.ipynb)

- [Multi-metric registration](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistration/MultiMetricRegistration.ipynb)

- [Image math operations](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/iMath_help.ipynb)

- [Wrapping ITK code](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/UsingITK.ipynb)

<br>

## References

The main references can be found at the main [ANTs](https://github.com/ANTsX/ANTs#boilerplate-ants) repo. A Google Scholar search also reveals plenty of explanation of methods and evaluation results by [the community](https://scholar.google.com/scholar?start=0&q=advanced+normalization+tools+ants+image+registration&hl=en&as_sdt=0,40) and by [ourselves](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C40&q=advanced+normalization+tools+ants+image+registration+-avants+-tustison&btnG=).
