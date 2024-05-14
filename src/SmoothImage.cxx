
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <exception>
#include <algorithm>
#include <vector>

#include "itkDiscreteGaussianImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;


template <typename ImageType>
AntsImage<ImageType> smoothImage( AntsImage<ImageType> & myPointer,
                                  std::vector<double> sigma,
                                  bool sigmaInPhysicalCoordinates,
                                  unsigned int kernelwidth)
{
  typename ImageType::Pointer inimg = myPointer.ptr;
  
  typedef itk::DiscreteGaussianImageFilter< ImageType, ImageType > discreteGaussianImageFilterType;
  typename discreteGaussianImageFilterType::Pointer filter = discreteGaussianImageFilterType::New();
  const unsigned int ImageDimension = ImageType::ImageDimension;
  if ( !sigmaInPhysicalCoordinates )
  {
    filter->SetUseImageSpacingOff();
  }
  else
  {
    filter->SetUseImageSpacingOn();
  }
  if ( sigma.size() == 1 )
  {
    filter->SetVariance( sigma[0] * sigma[0]);
  }
  else if ( sigma.size() == ImageDimension )
  {
    typename discreteGaussianImageFilterType::ArrayType varianceArray;
    for( unsigned int d = 0; d < ImageDimension; d++ )
    {
      varianceArray[d] = sigma[d] * sigma[d];
    }
    filter->SetVariance( varianceArray );
  }
  filter->SetMaximumKernelWidth(kernelwidth);
  filter->SetMaximumError( 0.01f );
  filter->SetInput( inimg );
  filter->Update();

  AntsImage<ImageType> ants_outimg = { filter->GetOutput() };
  return ants_outimg;
}

void local_SmoothImage(nb::module_ &m)
{
  m.def("SmoothImage", &smoothImage<itk::Image<float, 2>>);
  m.def("SmoothImage", &smoothImage<itk::Image<float, 3>>);
  m.def("SmoothImage", &smoothImage<itk::Image<float, 4>>);
}

