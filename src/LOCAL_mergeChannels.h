#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "LOCAL_antsImage.h"


template< class VectorImageType, class ImageType>
std::vector<AntsImage<ImageType>> splitChannels( AntsImage<VectorImageType> & antsimage );

template< class ImageType, class VectorImageType >
AntsImage<VectorImageType> mergeChannels( std::vector<AntsImage<ImageType>> imageList );