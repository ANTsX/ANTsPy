
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkConstantPadImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template < typename ImageType >
py::capsule padImage( py::capsule & antsImage, 
                      std::vector<int> lowerPadDims,
                      std::vector<int> upperPadDims,
                      float padValue )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkImage = as< ImageType >( antsImage );


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

  return wrap< ImageType >( padFilter->GetOutput() );
}

PYBIND11_MODULE(padImage, m)
{
  m.def("padImageF2", &padImage<itk::Image<float, 2>>);
  m.def("padImageF3", &padImage<itk::Image<float, 3>>);
  m.def("padImageF4", &padImage<itk::Image<float, 4>>);
}

