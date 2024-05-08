
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "itkImage.h"
#include "itkAntiAliasBinaryImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;


template <typename ImageType, unsigned int Dimension>
py::capsule antiAlias( py::capsule antsImage )
{
  typedef itk::Image<float, Dimension> FloatImageType;
  typedef typename FloatImageType::Pointer FloatImagePointerType;

  typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

  // Take the absolute value of the image
  typedef itk::AntiAliasBinaryImageFilter<ImageType, FloatImageType> AntiAliasBinaryImageFilterType;
  typename AntiAliasBinaryImageFilterType::Pointer antiAliasFilter = AntiAliasBinaryImageFilterType::New();
  antiAliasFilter->SetInput( itkImage );
  antiAliasFilter->Update();

  return wrap< FloatImageType >( antiAliasFilter->GetOutput() );
}


PYBIND11_MODULE(antiAlias, m)
{
  m.def("antiAliasUC2", &antiAlias<itk::Image<unsigned char, 2>, 2>);
  m.def("antiAliasUC3", &antiAlias<itk::Image<unsigned char, 3>, 3>);
}