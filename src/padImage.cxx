
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkConstantPadImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template < typename ImageType >
AntsImage<ImageType> padImage( AntsImage<ImageType> & antsImage,
                      std::vector<int> lowerPadDims,
                      std::vector<int> upperPadDims,
                      float padValue )
{
  typedef typename ImageType::Pointer ImagePointerType;
  typename ImageType::Pointer itkImage = antsImage.ptr;


  typename ImageType::SizeType lowerExtendRegion;
  lowerExtendRegion[0] = lowerPadDims[0];
  lowerExtendRegion[1] = lowerPadDims[1];
  if (lowerPadDims.size() == 3)
  {
    lowerExtendRegion[2] = lowerPadDims[2];
  }

  typename ImageType::SizeType upperExtendRegion;
  upperExtendRegion[0] = upperPadDims[0];
  upperExtendRegion[1] = upperPadDims[1];
  if (upperPadDims.size() == 3)
  {
    upperExtendRegion[2] = upperPadDims[2];
  }

  //ImageType::PixelType constantPixel = padValue;
  typedef itk::ConstantPadImageFilter<ImageType, ImageType> PadImageFilterType;
  typename PadImageFilterType::Pointer padFilter = PadImageFilterType::New();
  padFilter->SetInput( itkImage );
  padFilter->SetPadLowerBound( lowerExtendRegion );
  padFilter->SetPadUpperBound( upperExtendRegion );
  padFilter->SetConstant( padValue );
  padFilter->Update();
  FixNonZeroIndex<ImageType>( padFilter->GetOutput() );

  AntsImage<ImageType> myImage = { padFilter->GetOutput() };
  return myImage;
}

void local_padImage(nb::module_ &m) 
{
  m.def("padImage", &padImage<itk::Image<float, 2>>);
  m.def("padImage", &padImage<itk::Image<float, 3>>);
  m.def("padImage", &padImage<itk::Image<float, 4>>);
}

