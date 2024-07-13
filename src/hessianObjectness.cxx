
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

#include "itkHessianToObjectnessMeasureImageFilter.h"
#include "itkMultiScaleHessianBasedMeasureImageFilter.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkNumericTraits.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType>
AntsImage<ImageType> hessianObjectness( AntsImage<ImageType> & antsImage,
                                        unsigned int objectDimension,
                                        bool isBrightObject,
                                        float sigmaMin,
                                        float sigmaMax,
                                        unsigned int numberOfSigmaSteps,
                                        bool useSigmaLogarithmicSpacing,
                                        float alpha,
                                        float beta,
                                        float gamma,
                                        bool setScaleObjectnessMeasure )
{
  using PixelType = typename ImageType::PixelType;
  const unsigned int ImageDimension = ImageType::ImageDimension;
  using RealPixelType = typename itk::NumericTraits<PixelType>::RealType; 
  
  using HessianPixelType = itk::SymmetricSecondRankTensor<RealPixelType, ImageDimension>;
  using HessianImageType = itk::Image<HessianPixelType, ImageDimension>;
  using ObjectnessFilterType = itk::HessianToObjectnessMeasureImageFilter<HessianImageType, ImageType>;
  using MultiScaleEnhancementFilterType = itk::MultiScaleHessianBasedMeasureImageFilter<ImageType, HessianImageType, ImageType>;

  typename ImageType::Pointer itkImage = antsImage.ptr;

  typename ObjectnessFilterType::Pointer objectnessFilter = ObjectnessFilterType::New();
  objectnessFilter->SetScaleObjectnessMeasure( setScaleObjectnessMeasure );
  objectnessFilter->SetBrightObject( isBrightObject );
  objectnessFilter->SetAlpha( alpha );
  objectnessFilter->SetBeta( beta );
  objectnessFilter->SetGamma( gamma );
  objectnessFilter->SetObjectDimension( objectDimension );

  typename MultiScaleEnhancementFilterType::Pointer multiScaleEnhancementFilter = MultiScaleEnhancementFilterType::New();
  multiScaleEnhancementFilter->SetInput( itkImage );
  if( useSigmaLogarithmicSpacing )
    {
    multiScaleEnhancementFilter->SetSigmaStepMethodToLogarithmic();
    } 
  else 
    {
    multiScaleEnhancementFilter->SetSigmaStepMethodToEquispaced();
    }
  multiScaleEnhancementFilter->SetSigmaMinimum( sigmaMin  );
  multiScaleEnhancementFilter->SetSigmaMaximum( sigmaMax );
  multiScaleEnhancementFilter->SetNumberOfSigmaSteps( numberOfSigmaSteps );
  multiScaleEnhancementFilter->SetHessianToMeasureFilter( objectnessFilter );
  multiScaleEnhancementFilter->Update();

  AntsImage<ImageType> outImage = { multiScaleEnhancementFilter->GetOutput() };
  return outImage;
}

void local_hessianObjectness(nb::module_ &m)
{
  m.def("hessianObjectnessF2", &hessianObjectness<itk::Image<float, 2>>);
  m.def("hessianObjectnessF3", &hessianObjectness<itk::Image<float, 3>>);
}

