
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkHausdorffDistanceImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template<class PrecisionType, unsigned int ImageDimension>
nb::dict hausdorffDistance( AntsImage<itk::Image<PrecisionType, ImageDimension>> & antsImage1,
                               AntsImage<itk::Image<PrecisionType, ImageDimension>> & antsImage2 )
{
  using ImageType = itk::Image<PrecisionType, ImageDimension>;
  using ImagePointerType = typename ImageType::Pointer;

  ImagePointerType inputImage1 = antsImage1.ptr;
  ImagePointerType inputImage2 = antsImage2.ptr;

  using FilterType = itk::HausdorffDistanceImageFilter<ImageType, ImageType>;
  typename FilterType::Pointer hausdorff = FilterType::New();
  hausdorff->SetInput1( inputImage1 );
  hausdorff->SetInput2( inputImage2 );
  hausdorff->SetUseImageSpacing( true );
  hausdorff->Update();

  typename FilterType::RealType hausdorffDistance = hausdorff->GetHausdorffDistance();
  typename FilterType::RealType averageHausdorffDistance = hausdorff->GetAverageHausdorffDistance();

  nb::dict hausdorffDistances;
  hausdorffDistances["Distance"] = hausdorffDistance;
  hausdorffDistances["AverageDistance"] = averageHausdorffDistance;
  return (hausdorffDistances);
}

void local_hausdorffDistance(nb::module_ &m)
{
  m.def("hausdorffDistance2D", &hausdorffDistance<unsigned int, 2>);
  m.def("hausdorffDistance3D", &hausdorffDistance<unsigned int, 3>);
  m.def("hausdorffDistance4D", &hausdorffDistance<unsigned int, 4>);
}

