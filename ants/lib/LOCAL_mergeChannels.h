#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "LOCAL_antsImage.h"


template< class VectorImageType, class ImageType>
std::vector<py::capsule > splitChannels( py::capsule & antsimage );

template< class ImageType, class VectorImageType >
py::capsule mergeChannels( std::vector<void *> imageList );

template< class ImageType, class VectorImageType >
py::capsule mergeChannels2( std::vector<py::capsule> imageList );

template< class ImageType, class VectorImageType >
py::capsule mergeChannels3( std::vector<py::capsule> imageList );