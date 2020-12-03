
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkHistogramMatchingImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template < typename ImageType >
py::capsule histogramMatchImage( py::capsule & antsSourceImage,
                                 py::capsule & antsReferenceImage,
                                 unsigned int numberOfHistogramBins,
                                 unsigned int numberOfMatchPoints,
                                 bool useThresholdAtMeanIntensity )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkSourceImage = as< ImageType >( antsSourceImage );
  ImagePointerType itkReferenceImage = as< ImageType >( antsReferenceImage );

  typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetSourceImage( itkSourceImage );
  filter->SetReferenceImage( itkReferenceImage );
  filter->ThresholdAtMeanIntensityOff();
  if( useThresholdAtMeanIntensity )
    {
    filter->ThresholdAtMeanIntensityOn();
    }
  filter->SetNumberOfHistogramLevels( numberOfHistogramBins );
  filter->SetNumberOfMatchPoints( numberOfMatchPoints );
  filter->Update();

  return wrap< ImageType >( filter->GetOutput() );
}

PYBIND11_MODULE(histogramMatchImage, m)
{
  m.def("histogramMatchImageF2", &histogramMatchImage<itk::Image<float, 2>>);
  m.def("histogramMatchImageF3", &histogramMatchImage<itk::Image<float, 3>>);
  m.def("histogramMatchImageF4", &histogramMatchImage<itk::Image<float, 4>>);
}

