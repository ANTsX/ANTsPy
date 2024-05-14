
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "itkImage.h"
#include "itkAntiAliasBinaryImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType, unsigned int Dimension>
AntsImage<itk::Image<float, Dimension>> antiAlias( AntsImage<ImageType> & antsImage )
{
  typedef itk::Image<float, Dimension> FloatImageType;
  typedef typename FloatImageType::Pointer FloatImagePointerType;

  typename ImageType::Pointer itkImage = antsImage.ptr;

  // Take the absolute value of the image
  typedef itk::AntiAliasBinaryImageFilter<ImageType, FloatImageType> AntiAliasBinaryImageFilterType;
  typename AntiAliasBinaryImageFilterType::Pointer antiAliasFilter = AntiAliasBinaryImageFilterType::New();
  antiAliasFilter->SetInput( itkImage );
  antiAliasFilter->Update();

  AntsImage<FloatImageType> out_ants_image = { antiAliasFilter->GetOutput() };
  return out_ants_image;
}


void local_antiAlias(nb::module_ &m) {
  m.def("antiAliasUC2", &antiAlias<itk::Image<unsigned char, 2>, 2>);
  m.def("antiAliasUC3", &antiAlias<itk::Image<unsigned char, 3>, 3>);
}