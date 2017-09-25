#ifndef ANTSPYREADIMAGE_H
#define ANTSPYREADIMAGE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <tuple>
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itkPyBuffer.h"

#include "LOCAL_antsImage.h"
#include "LOCAL_antsImage.h"

template <typename ImageType>
py::capsule fromNumpy( py::array data, py::tuple datashape );


#endif // ANTSPYREADIMAGE_H