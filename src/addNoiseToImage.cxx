
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

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType>
AntsImage<ImageType> additiveGaussianNoise( AntsImage<ImageType> & antsImage,
                                   float mean,
                                   float standardDeviation )
{
  typename ImageType::Pointer itkImage = antsImage.ptr;
  using NoiseFilterType = itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetMean( mean );
  noiser->SetStandardDeviation( standardDeviation );
  noiser->Update();
  AntsImage<ImageType> outImage = { noiser->GetOutput() };
  return outImage;
}

template <typename ImageType>
AntsImage<ImageType> saltAndPepperNoise( AntsImage<ImageType> & antsImage,
                                float probability,
                                float saltValue,
                                float pepperValue )
{
  typename ImageType::Pointer itkImage = antsImage.ptr;
  using NoiseFilterType = itk::SaltAndPepperNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetProbability( probability );
  noiser->SetSaltValue( saltValue );
  noiser->SetPepperValue( pepperValue );
  noiser->Update();
  AntsImage<ImageType> outImage = { noiser->GetOutput() };
  return outImage;
}

template <typename ImageType>
AntsImage<ImageType> shotNoise( AntsImage<ImageType> & antsImage,
                       float scale
 )
{
  typename ImageType::Pointer itkImage = antsImage.ptr;
  using NoiseFilterType = itk::ShotNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetScale( scale );
  noiser->Update();

  AntsImage<ImageType> outImage = { noiser->GetOutput() };
  return outImage;
}

template <typename ImageType>
AntsImage<ImageType> speckleNoise( AntsImage<ImageType> & antsImage,
                          float scale
 )
{
  typename ImageType::Pointer itkImage = antsImage.ptr;
  using NoiseFilterType = itk::SpeckleNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetStandardDeviation( scale );
  noiser->Update();

  AntsImage<ImageType> outImage = { noiser->GetOutput() };
  return outImage;
}

void local_addNoiseToImage(nb::module_ &m)
{
  m.def("additiveGaussianNoiseF2", &additiveGaussianNoise<itk::Image<float, 2>>);
  m.def("additiveGaussianNoiseF3", &additiveGaussianNoise<itk::Image<float, 3>>);

  m.def("saltAndPepperNoiseF2", &saltAndPepperNoise<itk::Image<float, 2>>);
  m.def("saltAndPepperNoiseF3", &saltAndPepperNoise<itk::Image<float, 3>>);

  m.def("shotNoiseF2", &shotNoise<itk::Image<float, 2>>);
  m.def("shotNoiseF3", &shotNoise<itk::Image<float, 3>>);

  m.def("speckleNoiseF2", &speckleNoise<itk::Image<float, 2>>);
  m.def("speckleNoiseF3", &speckleNoise<itk::Image<float, 3>>);
}

