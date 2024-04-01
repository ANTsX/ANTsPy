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

#include "local_antsImage.h"

namespace nb = nanobind;

using namespace nb::literals;

using StrVector = std::vector<std::string>;


template <typename ImageType>
void * additiveGaussianNoiseHelper( typename ImageType::Pointer itkImage,
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

void * additiveGaussianNoise(void * ptr, std::string imageType,  float mean, float standardDeviation ) {
      if (imageType == "UC3") {
        using ImageType = itk::Image<unsigned char, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return additiveGaussianNoiseHelper<ImageType>( itkImage, mean, standardDeviation );
    }

    if (imageType == "UI3") {
        using ImageType = itk::Image<unsigned int, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return additiveGaussianNoiseHelper<ImageType>( itkImage,  mean, standardDeviation );
    }

    if (imageType == "F3") {
        using ImageType = itk::Image<float, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return additiveGaussianNoiseHelper<ImageType>( itkImage,  mean, standardDeviation );
    }

    if (imageType == "D3") {
        using ImageType = itk::Image<double, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return additiveGaussianNoiseHelper<ImageType>( itkImage, mean, standardDeviation );
    }
}

void local_addNoiseToImage(nb::module_ &m) {
    m.def("additiveGaussianNoise", &additiveGaussianNoise);
}
