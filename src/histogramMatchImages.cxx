
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
#include "itkHistogramMatchingImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template < typename ImageType >
AntsImage<ImageType> histogramMatchImage( AntsImage<ImageType> & antsSourceImage,
                                 AntsImage<ImageType> & antsReferenceImage,
                                 unsigned int numberOfHistogramBins,
                                 unsigned int numberOfMatchPoints,
                                 bool useThresholdAtMeanIntensity )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkSourceImage = antsSourceImage.ptr;
  ImagePointerType itkReferenceImage = antsReferenceImage.ptr;

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

  AntsImage<ImageType> out_ants_image = { filter->GetOutput()  };
  return out_ants_image;
}

void local_histogramMatchImages(nb::module_ &m)
{
  m.def("histogramMatchImageF2", &histogramMatchImage<itk::Image<float, 2>>);
  m.def("histogramMatchImageF3", &histogramMatchImage<itk::Image<float, 3>>);
  m.def("histogramMatchImageF4", &histogramMatchImage<itk::Image<float, 4>>);
}

