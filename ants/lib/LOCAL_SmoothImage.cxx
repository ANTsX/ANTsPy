
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <algorithm>
#include <vector>

#include "itkDiscreteGaussianImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;


template <typename ImageType>
py::capsule smoothImage( py::capsule ants_inimg,
                                  std::vector<double> sigma,
                                  bool sigmaInPhysicalCoordinates,
                                  unsigned int kernelwidth)
{
  typedef typename ImageType::Pointer ImagePointerType;
  typename ImageType::Pointer inimg = as< ImageType >( ants_inimg );
  
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

  py::capsule ants_outimg = wrap<ImageType>( filter->GetOutput() );
  return ants_outimg;
}

PYBIND11_MODULE(SmoothImage, m)
{
  m.def("SmoothImage2D", &smoothImage<itk::Image<float, 2>>);
  m.def("SmoothImage3D", &smoothImage<itk::Image<float, 3>>);
  m.def("SmoothImage4D", &smoothImage<itk::Image<float, 4>>);
}

