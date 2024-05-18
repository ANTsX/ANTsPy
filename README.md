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

The ANTsPy library wraps the well-established C++ biomedical image processing framework [ANTs](https://github.com/antsx/ants). It includes blazing-fast reading and writing of medical images, algorithms for registration, segmentation, and statistical learning, as well as functions to create publication-ready visualizations.

If you are looking to train deep learning models on medical imaging datasets, you might be interested in [ANTsPyNet](https://github.com/antsx/antspy) which provides tools for training and visualizing deep learning models.

<br>

## Installation

### Pre-compiled binaries

The easiest way to install ANTsPy is via the latest pre-compiled binaries from PyPI.

```bash
pip install antspyx
```

Because of limited storage space, pip binaries are not available for every combination of python
version and platform. If we do not have releases for your platform on PyPI, you can check the
[Releases](https://github.com/antsx/antspy/releases) page for archived binaries.

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

### Reading and writing images

You can read and write images of any format.

```python
import ants
img = ants.image_read('path/to/image.nii.gz')
ants.image_write(img, 'path/to/image.nii.gz')
```

Printing to the console provides a great deal of metadata information about the image.

```python
print(img)
```

```
ANTsImage
         Pixel Type : float (float32)
         Components : 1
         Dimensions : (256, 256)
         Spacing    : (1.0, 1.0)
         Origin     : (0.0, 0.0)
         Direction  : [1. 0. 0. 1.]
```

### Image operations

#### Basic

Images can be operated on similar to numpy arrays - e.g., all math operations will work as expected.

```python
img = ants.image_read(ants.get_data('r16'))
img2 = ants.image_read(ants.get_data('r64'))

img + img2
img - img2
img * img2
img / img2
img ** 2
```

#### Indexing 

You can also index images as you would a numpy array. Where possible, indexing an image will return an image with metadata intact.

```python
img = ants.image_read(ants.get_data('mni')) # 3D image

img[:20,:20,:20] # 3D image
img[:,:,20] # 2D image
img[:,20,20] # 1D array
img[20,20,20] # sigle value

# setting works as well
img[:20,:20,:20] = 10
```

#### Advanced

There is a large collection of advanced image operations that can be performed on images. 

```python
img = ants.image_read(ants.get_data('mni')) # 3D image
img = ants.smooth_image(img, 2)
img = ants.resample_image(img, (3,3,3))
img = ants.pad_image(img, pad_width=(4,4,4))

# chaining operations is possible
img = img.smooth_image(2).resample_image((3,3,3)).pad_image(pad_width=(4,4,4))
```

And if you ever need to convert to or from numpy, it is straight-forward to do so.

```python
img = ants.image_read(ants.get_data('mni')) # 3D image
arr = img.numpy()
arr += 2
img2 = ants.from_numpy(arr)
```

### Segmentation

Atropos is an example of a powerful three-class segmentation algorithm provided to you.

```python
img = ants.image_read(ants.get_data("r16"))
mask = ants.get_mask(img)
result = ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask)
```

### Registration

The full registration functionality of ANTs is available via the `ants.registration` function.

```python
fixed_image = ants.image_read(ants.get_ants_data('r16')).resample_image((60,60), 1, 0)
moving_image = ants.image_read(ants.get_ants_data('r64')).resample_image((60,60), 1, 0)
mytx = ants.registration(fixed_image, moving_image, type_of_transform = 'SyN' )
```

### Plotting

A diverse set of functions are available to flexibly visualize images, optionally with discrete or continuous overlays. The `ants.plot` function will meet most needs.

```python
img = ants.image_read(ants.get_data("mni")) # 3D image
ants.plot(img)

# with overlay
ants.plot(img, overlay = img > img.mean())
```

<br>

## Tutorials

Resources for learning about ANTsPy can be found in the [tutorials](https://github.com/ANTsX/ANTsPy/tree/master/tutorials) folder. A selection of especially useful tutorials is presented below.

- Basic overview [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/tutorial_5min.md)]
- Composite registrations [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistrations.ipynb)]
- Multi-metric registration [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/concatenateRegistration/MultiMetricRegistration.ipynb)]
- Image math operations [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/iMath_help.ipynb)]
- Wrapping ITK code [[Link](https://github.com/ANTsX/ANTsPy/blob/master/tutorials/UsingITK.ipynb)]

More tutorials can be found in the [ANTs](https://github.com/ANTsX/ANTs) repository.

<br>

## Contributing

If you have a question, feature request, or bug report the best way to get help is by posting an issue on the GitHub page. We welcome any new contributions and ideas. If you want to add code, the best way to get started is by reading the [contributors guide](https://github.com/ANTsX/ANTsPy/blob/master/CONTRIBUTING.md) that runs through the structure of the project and how we go about wrapping ITK and ANTs code in C++.

<br>

## References

The main references can be found at the main [ANTs](https://github.com/ANTsX/ANTs#boilerplate-ants) repo. A Google Scholar search also reveals plenty of explanation of methods and evaluation results by [the community](https://scholar.google.com/scholar?start=0&q=advanced+normalization+tools+ants+image+registration&hl=en&as_sdt=0,40) and by [ourselves](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C40&q=advanced+normalization+tools+ants+image+registration+-avants+-tustison&btnG=).
