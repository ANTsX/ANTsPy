
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkHausdorffDistanceImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;
using namespace py::literals;

template<class PrecisionType, unsigned int ImageDimension>
py::capsule hausdorffDistance( py::capsule & antsImage1,
                               py::capsule & antsImage2 )
{
  using ImageType = itk::Image<PrecisionType, ImageDimension>;
  using ImagePointerType = typename ImageType::Pointer;

  ImagePointerType inputImage1 = as< ImageType >( antsImage1 );
  ImagePointerType inputImage2 = as< ImageType >( antsImage2 );

  using FilterType = itk::HausdorffDistanceImageFilter<ImageType, ImageType>;
  typename FilterType::Pointer hausdorff = FilterType::New();
  hausdorff->SetInput1( inputImage1 );
  hausdorff->SetInput2( inputImage2 );
  hausdorff->SetUseImageSpacing( true );
  hausdorff->Update();

  typename FilterType::RealType hausdorffDistance = hausdorff->GetHausdorffDistance();
  typename FilterType::RealType averageHausdorffDistance = hausdorff->GetAverageHausdorffDistance();

  py::dict hausdorffDistances = py::dict( "Distance"_a=hausdorffDistance,
                                          "AverageDistance"_a=averageHausdorffDistance );
  return (hausdorffDistances);
}

PYBIND11_MODULE(hausdorffDistance, m)
{
  m.def("hausdorffDistance2D", &hausdorffDistance<unsigned int, 2>);
  m.def("hausdorffDistance3D", &hausdorffDistance<unsigned int, 3>);
  m.def("hausdorffDistance4D", &hausdorffDistance<unsigned int, 4>);
}

