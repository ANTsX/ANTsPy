
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>
#include <algorithm>

#include "itkImage.h"

#include "antscore/itkSurfaceImageCurvature.h"

#include "LOCAL_antsImage.h"


namespace py = pybind11;

py::capsule weingartenImageCurvature( py::capsule myimage,
                                      float sigma, int opt )
{
  typedef itk::Image<float, 3> ImageType;

  typedef typename ImageType::Pointer       ImagePointerType;
  enum { ImageDimension = ImageType::ImageDimension };
  typedef itk::SurfaceImageCurvature<ImageType> ParamType;
  typename ParamType::Pointer Parameterizer = ParamType::New();
  typename ImageType::Pointer input = as<ImageType>( myimage );
  typename ImageType::DirectionType imgdir = input->GetDirection();
  typename ImageType::DirectionType iddir = input->GetDirection();
  iddir.SetIdentity();
  input->SetDirection( iddir );
  Parameterizer->SetInputImage(input);
  Parameterizer->SetNeighborhoodRadius( 1. );
  if( sigma <= 0.5 )
      {
      sigma = 1.66;
      }
  Parameterizer->SetSigma(sigma);
  Parameterizer->SetUseLabel(false);
  Parameterizer->SetUseGeodesicNeighborhood(false);
  float sign = 1.0;
  Parameterizer->SetkSign(sign);
  Parameterizer->SetThreshold(0);
  if( opt != 5 && opt != 6 )
      {
      Parameterizer->ComputeFrameOverDomain( 3 );
      }
  else
      {
      Parameterizer->ComputeFrameOverDomain( opt );
      }
  typename ImageType::Pointer output = Parameterizer->GetFunctionImage();
  output->SetDirection( imgdir );
  return wrap<ImageType>( output );

}


PYBIND11_MODULE(weingartenImageCurvature, m)
{
  m.def("weingartenImageCurvature", &weingartenImageCurvature);
}