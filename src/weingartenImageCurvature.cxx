
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
#include <algorithm>

#include "itkImage.h"

#include "antscore/itkSurfaceImageCurvature.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

AntsImage<itk::Image<float, 3>> weingartenImageCurvature( AntsImage<itk::Image<float, 3>> myimage,
                                      float sigma, int opt )
{
  typedef itk::Image<float, 3> ImageType;

  typedef typename ImageType::Pointer       ImagePointerType;
  enum { ImageDimension = ImageType::ImageDimension };
  typedef itk::SurfaceImageCurvature<ImageType> ParamType;
  typename ParamType::Pointer Parameterizer = ParamType::New();
  typename ImageType::Pointer input = myimage.ptr;
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

  AntsImage<ImageType> outImage = { output };
  return outImage;

}


void local_weingartenImageCurvature(nb::module_ &m)
{
  m.def("weingartenImageCurvature", &weingartenImageCurvature);
}