
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
//#include <Rcpp.h>

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;


template <unsigned int Dimension>
double antsImageMutualInformation( AntsImage<itk::Image< float , Dimension >> & in_image1, 
                                   AntsImage<itk::Image< float , Dimension >> & in_image2 )
{
  double mi = 1;
  typedef itk::Image< float , Dimension > ImageType;
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkImage1 = in_image1.ptr;
  ImagePointerType itkImage2 = in_image2.ptr;

  typedef itk::MattesMutualInformationImageToImageMetricv4
    <ImageType, ImageType, ImageType> MetricType;
  unsigned int bins = 32;
  typename MetricType::Pointer metric = MetricType::New();
  metric->SetFixedImage( itkImage1 );
  metric->SetMovingImage( itkImage2 );
  metric->SetNumberOfHistogramBins( bins );
  metric->Initialize();
  mi = metric->GetValue();
  return mi;

}

void local_antsImageMutualInformation(nb::module_ &m)
{
  m.def("antsImageMutualInformation2D", &antsImageMutualInformation<2>);
  m.def("antsImageMutualInformation3D", &antsImageMutualInformation<3>);
  m.def("antsImageMutualInformation4D", &antsImageMutualInformation<4>);
}