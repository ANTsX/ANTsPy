
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"

#include "itkAdditiveGaussianNoiseImageFilter.h"
#include "itkSaltAndPepperNoiseImageFilter.h"
#include "itkShotNoiseImageFilter.h"
#include "itkSpeckleNoiseImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template <typename ImageType>
py::capsule additiveGaussianNoise( py::capsule & antsImage,
                                   float mean,
                                   float standardDeviation )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkImage = as<ImageType>( antsImage );

  using NoiseFilterType = itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetMean( mean );
  noiser->SetStandardDeviation( standardDeviation );
  noiser->Update();

  return wrap<ImageType>( noiser->GetOutput() );
}

template <typename ImageType>
py::capsule saltAndPepperNoise( py::capsule & antsImage,
                                float probability,
                                float saltValue,
                                float pepperValue )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkImage = as<ImageType>( antsImage );

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
py::capsule shotNoise( py::capsule & antsImage,
                       float scale
 )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkImage = as<ImageType>( antsImage );

  using NoiseFilterType = itk::ShotNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetScale( scale );
  noiser->Update();

  return wrap<ImageType>( noiser->GetOutput() );
}

template <typename ImageType>
py::capsule speckleNoise( py::capsule & antsImage,
                          float scale
 )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkImage = as<ImageType>( antsImage );

  using NoiseFilterType = itk::SpeckleNoiseImageFilter<ImageType, ImageType>;
  typename NoiseFilterType::Pointer noiser = NoiseFilterType::New();
  noiser->SetInput( itkImage );
  noiser->SetStandardDeviation( scale );
  noiser->Update();

  return wrap<ImageType>( noiser->GetOutput() );
}

PYBIND11_MODULE(addNoiseToImage, m)
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

