# Advanced Normalization Tools in Python

[![Coverage Status](https://coveralls.io/repos/github/ANTsX/ANTsPy/badge.svg?branch=master)](https://coveralls.io/github/ANTsX/ANTsPy?branch=master)
<a href='http://antspyx.readthedocs.io/en/latest/?badge=latest'>
</a>
[![PyPI - Downloads](https://img.shields.io/pypi/dm/antspyx?label=pypi%20downloads)](https://pypi.org/project/antspyx/)
[![Nightly Build](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/wheels.yml)
[![ci-pytest](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-pytest.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-pytest.yml)
[![ci-docker](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml/badge.svg)](https://github.com/ANTsX/ANTsPy/actions/workflows/ci-docker.yml)
[![docs](https://readthedocs.org/projects/antspy/badge/?version=latest&style=flat)](https://antspy.readthedocs.io/en/latest/)

[![Docker Pulls](https://img.shields.io/docker/pulls/antsx/antspy.svg)](https://hub.docker.com/repository/docker/antsx/antspy)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)
[![PubMed](https://img.shields.io/badge/ANTsX_paper-Open_Access-8DABFF?logo=pubmed)](https://pubmed.ncbi.nlm.nih.gov/33907199/)

The ANTsPy library wraps the well-established C++ biomedical image processing framework [ANTs](https://github.com/antsx/ants). It includes blazing-fast reading and writing of medical images, algorithms for registration, segmentation, and statistical learning, as well as functions to create publication-ready visualizations.

If you are looking to train deep learning models on medical imaging datasets, you might be interested in [ANTsPyNet](https://github.com/ANTsX/ANTsPyNet) which provides tools for training and visualizing deep learning models.

<br>

## Installation

### Pre-compiled binaries

The easiest way to install ANTsPy is via the latest pre-compiled binaries from PyPI.

```bash
pip install antspyx
```
Or alternatively from conda:

```bash
conda install conda-forge::antspyx
```

Because of limited storage space, pip binaries are not available for every combination of python
version and platform. We also have had to delete older releases to make space. If you
cannot find a binary you need on PyPI, you can check the
[Releases](https://github.com/antsx/antspy/releases) page for archived binaries.

Some Mac OS Python installations have compatibility issues with the pre-compiled
binaries. This means pip will not install binaries targeted for the current Mac OS
version, and will instead try to compile from source. The compatibility checks can be
disabled by setting the  environment variable `SYSTEM_VERSION_COMPAT=0`. More details on
the [wiki](https://github.com/ANTsX/ANTsPy/wiki/MacOS-wheel-compatibility-issues).

Windows users will need a compatible [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022) installation.


### Building from source

In some scenarios, it can make sense to build from source. In general, you can build ANTsPy as you would any other Python package.

```
git clone https://github.com/antsx/antspy
cd antspy
python -m pip install .
```

Further details about installing ANTsPy or building it from source can be found in the
[Installation Tutorial](https://github.com/antsx/antspy/blob/master/tutorials/Installation.md).

<br>

## Quickstart

Here is a basic overview of some of the things you can do with ANTsPy. The main functionality includes reading / writing images, basic and advanced image operations, segmentation, registration, and visualization.

```python
import ants

# read / write images
img = ants.image_read('path/to/image.nii.gz')
ants.image_write(img, 'path/to/image.nii.gz')

# basic operations
img + img2
img - img2
img[:20,:20,:20] # indexing returns an image

# advanced operations
img = ants.smooth_image(img, 2)
img = ants.resample_image(img, (3,3,3))
img.smooth_image(2).resample_image((3,3,3)) # chaining

# convert to or from numpy
arr = img.numpy()
img2 = ants.from_numpy(arr * 2)

# segmentation
result = ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=ants.get_mask(img))

# registration
result = ants.registration(fixed_image, moving_image, type_of_transform = 'SyN' )

# plotting
ants.plot(img, overlay = img > img.mean())
```

<br>

## Tutorials

Resources for learning about ANTsPy can be found in the [tutorials](https://github.com/ANTsX/ANTsPy/tree/master/tutorials) folder. A selection of especially useful tutorials is presented below.

- Basic overview [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/tutorial_5min.md)]
- Composite registrations [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistrations.ipynb)]
- Multi-metric registration [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/MultiMetricRegistration.ipynb)]
- Image math operations [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/iMath_help.md)]
- Wrapping ITK code [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/UsingITK.md)]

More tutorials can be found in the [ANTs](https://github.com/ANTsX/ANTs) repository.

<br>

## Contributing

If you have a question or bug report the best way to get help is by posting an issue on the GitHub page. We welcome any new contributions and ideas. If you want to add code, the best way to get started is by reading the [contributors guide](https://github.com/ANTsX/ANTsPy/blob/master/CONTRIBUTING.md) that runs through the structure of the project and how we go about wrapping ITK and ANTs code in C++.

You can support our work by starring the repository, citing our methods when relevant, or suggesting new features in the issues tab. These actions help increase the project's visibility and community reach.

<br>

## References

The main references can be found at the main [ANTs](https://github.com/ANTsX/ANTs#boilerplate-ants) repo. A Google Scholar search also reveals plenty of explanation of methods and evaluation results by [the community](https://scholar.google.com/scholar?start=0&q=advanced+normalization+tools+ants+image+registration&hl=en&as_sdt=0,40) and by [ourselves](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C40&q=advanced+normalization+tools+ants+image+registration+-avants+-tustison&btnG=).
