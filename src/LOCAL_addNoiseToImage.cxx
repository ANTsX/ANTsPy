
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

#include "itkAdditiveGaussianNoiseImageFilter.h"
#include "itkSaltAndPepperNoiseImageFilter.h"
#include "itkShotNoiseImageFilter.h"
#include "itkSpeckleNoiseImageFilter.h"

#include "LOCAL_antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType>
void * additiveGaussianNoise( typename ImageType::Pointer itkImage,
                                   float mean,
                                   float standardDeviation )
{
  using NoiseFilterType = itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetMean( mean );
  noiser->SetStandardDeviation( standardDeviation );
  noiser->Update();

  return wrap<ImageType>( noiser->GetOutput() );
}

template <typename ImageType>
void * saltAndPepperNoise( typename ImageType::Pointer itkImage,
                                float probability,
                                float saltValue,
                                float pepperValue )
{
  using NoiseFilterType = itk::SaltAndPepperNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetProbability( probability );
  noiser->SetSaltValue( saltValue );
  noiser->SetPepperValue( pepperValue );
  noiser->Update();

  return wrap<ImageType>( noiser->GetOutput() );
}

template <typename ImageType>
void * shotNoise( typename ImageType::Pointer itkImage,
                       float scale
 )
{
  using NoiseFilterType = itk::ShotNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetScale( scale );
  noiser->Update();

  return wrap<ImageType>( noiser->GetOutput() );
}

template <typename ImageType>
void * speckleNoise( typename ImageType::Pointer itkImage,
                          float scale
 )
{

  using NoiseFilterType = itk::SpeckleNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetStandardDeviation( scale );
  noiser->Update();

  return wrap<ImageType>( noiser->GetOutput() );
}

void local_addNoiseToImage(nb::module_ &m)
{
  m.def("additiveGaussianNoise", &additiveGaussianNoise<itk::Image<float, 2>>);
  m.def("additiveGaussianNoise", &additiveGaussianNoise<itk::Image<float, 3>>);

  m.def("saltAndPepperNoise", &saltAndPepperNoise<itk::Image<float, 2>>);
  m.def("saltAndPepperNoise", &saltAndPepperNoise<itk::Image<float, 3>>);

  m.def("shotNoise", &shotNoise<itk::Image<float, 2>>);
  m.def("shotNoise", &shotNoise<itk::Image<float, 3>>);

  m.def("speckleNoise", &speckleNoise<itk::Image<float, 2>>);
  m.def("speckleNoise", &speckleNoise<itk::Image<float, 3>>);
}

