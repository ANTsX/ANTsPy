#ifndef ANTSPYREADIMAGE_H
#define ANTSPYREADIMAGE_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <tuple>
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itkPyBuffer.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType>
AntsImage<ImageType> fromNumpy( nb::ndarray<nb::numpy> data, nb::tuple datashape );


#endif // ANTSPYREADIMAGE_H