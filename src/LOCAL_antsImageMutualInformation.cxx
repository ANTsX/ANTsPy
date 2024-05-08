
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>
//#include <Rcpp.h>

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;


template <unsigned int Dimension>
double antsImageMutualInformation( py::capsule in_image1, 
                                   py::capsule in_image2 )
{
  double mi = 1;
  typedef itk::Image< float , Dimension > ImageType;
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkImage1 = as<ImageType>( in_image1 );
  ImagePointerType itkImage2 = as<ImageType>( in_image2 );

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


PYBIND11_MODULE(antsImageMutualInformation, m)
{
  m.def("antsImageMutualInformation2D", &antsImageMutualInformation<2>);
  m.def("antsImageMutualInformation3D", &antsImageMutualInformation<3>);
  m.def("antsImageMutualInformation4D", &antsImageMutualInformation<4>);
}