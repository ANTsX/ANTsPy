
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkNormalizeImageFilter.h"
#include "itkSigmoidImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkInterpolateImageFunction.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"
#include "itkAnisotropicDiffusionFunction.h"
#include "itkAnisotropicDiffusionImageFilter.h"
#include "itkAnisotropicDiffusionImageFilter.hxx"
#include "itkGradientAnisotropicDiffusionImageFilter.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkScaleTransform.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

/*
Rotate a 2D image
See pg 174 of ITK User Guide Part 2

template <typename ImageType>
py::capsule rotateAntsImage2D( py::capsule & antsImage )
{

}
*/


template <typename ImageType>
std::vector< py::capsule > multiResolutionAntsImage( py::capsule & antsImage , unsigned int numberOfLevels )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );
    //unsigned int numberOfLevels = 4;

    typedef itk::RecursiveMultiResolutionPyramidImageFilter<ImageType, ImageType>  
        RecursiveMultiResolutionPyramidImageFilterType;
    
    typename RecursiveMultiResolutionPyramidImageFilterType::Pointer recursiveMultiResolutionPyramidImageFilter =
        RecursiveMultiResolutionPyramidImageFilterType::New();
    
    recursiveMultiResolutionPyramidImageFilter->SetInput(itkImage);
    recursiveMultiResolutionPyramidImageFilter->SetNumberOfLevels(numberOfLevels);
    recursiveMultiResolutionPyramidImageFilter->Update();

    std::vector< py::capsule > newImageList(numberOfLevels);
    // This outputs the levels (0 is the lowest resolution)
    for(unsigned int i = 0; i < numberOfLevels; ++i)
    {
        newImageList[i] = wrap< ImageType >( recursiveMultiResolutionPyramidImageFilter->GetOutput(i) );
    }
 
  return newImageList;
}

/*
Apply gaussian filter to ANTsImage with given variance and width parameters
*/
template <typename ImageType>
py::capsule blurAntsImage( py::capsule & antsImage, 
                            float variance, int maxKernelWidth )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

    typedef itk::DiscreteGaussianImageFilter< ImageType, ImageType > FilterType;
    typename FilterType::Pointer filter = FilterType::New();
    filter->SetInput( itkImage );

    filter->SetVariance( variance );
    filter->SetMaximumKernelWidth( maxKernelWidth );

    filter->Update();
    return wrap< ImageType >( filter->GetOutput() );
}

/*
Apply gaussian filter to ANTsImage with given variance and width parameters
*/
template <typename ImageType>
py::capsule locallyBlurAntsImage( py::capsule & antsImage, unsigned long nIterations, 
                                double conductance )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );
  
  typedef itk::GradientAnisotropicDiffusionImageFilter< ImageType, ImageType >FilterType;
  typedef typename FilterType::TimeStepType             TimeStepType;

  // Select time step size.
  TimeStepType spacingsize = 0;
  for( unsigned int d = 0; d < ImageType::ImageDimension; d++ )
    {
    TimeStepType sp = itkImage->GetSpacing()[d];
    spacingsize += sp * sp;
    }
  spacingsize = sqrt( spacingsize );

  // FIXME - cite reason for this step
  double dimPlusOne = ImageType::ImageDimension + 1;
  TimeStepType mytimestep = spacingsize / std::pow( 2.0 , dimPlusOne );
  TimeStepType reftimestep = 0.4 / std::pow( 2.0 , dimPlusOne );
  if ( mytimestep > reftimestep )
    {
    mytimestep = reftimestep;
    }

  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput( itkImage );
  filter->SetConductanceParameter( conductance ); // might need to change this
  filter->SetNumberOfIterations( nIterations );
  filter->SetTimeStep( mytimestep );

  filter->Update();
  //return filter->GetOutput();
    return wrap< ImageType >( filter->GetOutput() );
}


/*
Cast an ANTsImage to another pixel type
*/
template <typename InputImageType, typename OutputImageType>
py::capsule castAntsImage( py::capsule & antsImage )
{
    typename InputImageType::Pointer itkImage = as< InputImageType >( antsImage );

    typedef itk::CastImageFilter<InputImageType, OutputImageType > CastFilterType;
    typename CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput( itkImage );

    castFilter->Update();
    return wrap< OutputImageType >( castFilter->GetOutput() );
}


/*
Rescale the intensity of an ANTsImage linearly between a given minimum and maximum value
*/
template <typename ImageType>
py::capsule rescaleAntsImage( py::capsule & antsImage, float outputMinimum, float outputMaximum )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

    typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput( itkImage );
    rescaleFilter->SetOutputMinimum( outputMinimum );
    rescaleFilter->SetOutputMaximum( outputMaximum );
    
    rescaleFilter->Update();
    return wrap< ImageType >( rescaleFilter->GetOutput() );
}

/*
Shift the intensity of an ANTsImage then scale it
*/
template <typename ImageType>
py::capsule shiftScaleAntsImage( py::capsule & antsImage, float outputScale, float outputShift )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

    typedef itk::ShiftScaleImageFilter< ImageType, ImageType > ShiftScaleFilterType;
    typename ShiftScaleFilterType::Pointer shiftFilter = ShiftScaleFilterType::New();
    shiftFilter->SetInput( itkImage );
    shiftFilter->SetScale( outputScale );
    shiftFilter->SetShift( outputShift );
    
    shiftFilter->Update();
    return wrap< ImageType >( shiftFilter->GetOutput() );
}

/*
Normalize the intensity of an ANTsImage by setting its mean to zero and variance to one.
*/
template <typename ImageType>
py::capsule normalizeAntsImage( py::capsule & antsImage )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

    typedef itk::NormalizeImageFilter< ImageType, ImageType > NormalizeFilterType;
    typename NormalizeFilterType::Pointer normalizeFilter = NormalizeFilterType::New();
    normalizeFilter->SetInput( itkImage );

    normalizeFilter->Update();
    return wrap< ImageType >( normalizeFilter->GetOutput() );
}

/*
Apply Sigmoid filter to ANTsImage with given alpha and gamma parameters
*/
template <typename ImageType>
py::capsule sigmoidAntsImage( py::capsule & antsImage, 
                               float outputMinimum, float outputMaximum,
                               float alpha, float beta )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

    typedef itk::SigmoidImageFilter< ImageType, ImageType > SigmoidFilterType;
    typename SigmoidFilterType::Pointer sigmoidFilter = SigmoidFilterType::New();
    sigmoidFilter->SetInput( itkImage );
    sigmoidFilter->SetOutputMinimum( outputMinimum );
    sigmoidFilter->SetOutputMaximum( outputMaximum );
    sigmoidFilter->SetAlpha( alpha );
    sigmoidFilter->SetBeta( beta );

    sigmoidFilter->Update();
    return wrap< ImageType >( sigmoidFilter->GetOutput() );
}


/*
Filp an ANTsImage in a given direction
*/
template <typename ImageType>
py::capsule flipAntsImage( py::capsule & antsImage, unsigned int axis1, unsigned int axis2 )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

    typedef itk::FlipImageFilter< ImageType > FlipFilterType;
    typename FlipFilterType::Pointer flipFilter = FlipFilterType::New();
    flipFilter->SetInput( itkImage );

    typedef typename FlipFilterType::FlipAxesArrayType FlipAxesArrayType;
    FlipAxesArrayType flipArray;
    flipArray[0] = axis1;
    flipArray[1] = axis2;
    flipFilter->SetFlipAxes( flipArray );

    flipFilter->Update();
    return wrap< ImageType >( flipFilter->GetOutput() );
}

/*
Apply a translation to an ANTsImage. This function uses itk::TranslationTransform
which is faster than using itk::AffineTransform.

General steps for applying an ITK transform to an ANTsImage:
- unwrap py::capsule(s)
- create new transform smartpointer
- set transform parameters, etc
- create resample filter
- create interpolation filter
- set resample filter parameters, etc
*/
template <typename ImageType, typename InterpolatorType, typename PrecisionType, unsigned int Dimension>
py::capsule translateAntsImage( py::capsule & inputAntsImage, py::capsule refAntsImage, std::vector<float> translationList)
{
    // unwrap ANTsImage(s)
    typename ImageType::Pointer inputImage = as< ImageType >( inputAntsImage );
    typename ImageType::Pointer refImage = as< ImageType >( refAntsImage );

    // create new transform smartpointer
    typedef itk::TranslationTransform<PrecisionType,Dimension> TranslationTransformType;
    typename TranslationTransformType::Pointer translationTransform = TranslationTransformType::New();

    // set transform parameters, etc
    typename TranslationTransformType::OutputVectorType translation;
    for (unsigned int i = 0; i < translationList.size(); ++i )
    {
        translation[i] = translationList[i];
    }
    translationTransform->Translate(translation);

    // create resample filter
    typedef itk::ResampleImageFilter<ImageType,ImageType,PrecisionType,PrecisionType> ResampleFilterType;
    typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    // create interpolation filter
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

    // set resample filter parameters, etc
    resampleFilter->SetInput( inputImage );
    resampleFilter->SetSize( refImage->GetLargestPossibleRegion().GetSize() );
    resampleFilter->SetOutputSpacing( refImage->GetSpacing() );
    resampleFilter->SetOutputOrigin( refImage->GetOrigin() );
    resampleFilter->SetOutputDirection( refImage->GetDirection() );
    resampleFilter->SetInterpolator( interpolator );

    resampleFilter->SetTransform( translationTransform );
    resampleFilter->Update();

    return wrap< ImageType >( resampleFilter->GetOutput() );
}


/*
Scale an ANTsImage
*/
template <typename ImageType, typename InterpolatorType, typename PrecisionType, unsigned int Dimension>
py::capsule scaleAntsImage( py::capsule & antsImage, py::capsule & antsRefImage, std::vector<float> scaleList )
{
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );
    typename ImageType::Pointer refImage = as< ImageType >( antsRefImage );

    typedef itk::ScaleTransform<PrecisionType, Dimension> TransformType;
    typename TransformType::Pointer transform = TransformType::New();

    itk::FixedArray<PrecisionType, Dimension> scale;
    for (unsigned int i = 0; i < Dimension; i++ )
    {
        scale[i] = scaleList[i];
    }
    transform->SetScale(scale);

    itk::Point<float,Dimension> center;
    for (unsigned int i = 0; i < Dimension; i++ )
    {
        center[i] = itkImage->GetOrigin()[i] + itkImage->GetSpacing()[i] * itkImage->GetLargestPossibleRegion().GetSize()[i]/2;
    } 
    transform->SetCenter(center);

    // create resample filter
    typedef itk::ResampleImageFilter<ImageType,ImageType,PrecisionType,PrecisionType> ResampleFilterType;
    typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    // create interpolation filter
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

    // set resample filter parameters, etc
    resampleFilter->SetInput( itkImage );
    resampleFilter->SetSize( refImage->GetLargestPossibleRegion().GetSize() );
    resampleFilter->SetOutputSpacing( refImage->GetSpacing() );
    resampleFilter->SetOutputOrigin( refImage->GetOrigin() );
    resampleFilter->SetOutputDirection( refImage->GetDirection() );
    resampleFilter->SetInterpolator( interpolator );

    resampleFilter->SetTransform( transform );
    resampleFilter->Update();

    return wrap< ImageType >( resampleFilter->GetOutput() );
}


PYBIND11_MODULE(antsImageAugment, m)
{
    m.def("multiResolutionAntsImageF2", &multiResolutionAntsImage<itk::Image<float,2>>);
    m.def("multiResolutionAntsImageF3", &multiResolutionAntsImage<itk::Image<float,3>>);

    m.def("blurAntsImageF2", &blurAntsImage<itk::Image<double,2>>);
    m.def("blurAntsImageF3", &blurAntsImage<itk::Image<float,3>>);

    m.def("locallyBlurAntsImageF2", &locallyBlurAntsImage<itk::Image<float,2>>);
    m.def("locallyBlurAntsImageF3", &locallyBlurAntsImage<itk::Image<float,3>>);

    m.def("castAntsImageUC2F2", &castAntsImage<itk::Image<unsigned char,2>,itk::Image<float,2>>);
    m.def("castAntsImageUI2F2", &castAntsImage<itk::Image<unsigned int, 2>,itk::Image<float,2>>);
    m.def("castAntsImageD2F2", &castAntsImage<itk::Image<double,2>,itk::Image<float,2>>);

    m.def("castAntsImageUC3F3", &castAntsImage<itk::Image<unsigned char,3>,itk::Image<float,3>>);
    m.def("castAntsImageUI3F3", &castAntsImage<itk::Image<unsigned int, 3>,itk::Image<float,3>>);
    m.def("castAntsImageD3F3", &castAntsImage<itk::Image<double,3>,itk::Image<float,3>>);

    m.def("rescaleAntsImageF2", &rescaleAntsImage<itk::Image<float,2>>);
    m.def("rescaleAntsImageF3", &rescaleAntsImage<itk::Image<float,3>>);

    m.def("shiftScaleAntsImageF2", &shiftScaleAntsImage<itk::Image<float,2>>);
    m.def("shiftScaleAntsImageF3", &shiftScaleAntsImage<itk::Image<float,3>>);

    m.def("normalizeAntsImageF2", &normalizeAntsImage<itk::Image<float,2>>);
    m.def("normalizeAntsImageF3", &normalizeAntsImage<itk::Image<float,3>>);

    m.def("sigmoidAntsImageF2", &sigmoidAntsImage<itk::Image<float,2>>);
    m.def("sigmoidAntsImageF3", &sigmoidAntsImage<itk::Image<float,3>>);

    m.def("flipAntsImageF2", &flipAntsImage<itk::Image<float,2>>);
    m.def("flipAntsImageF3", &flipAntsImage<itk::Image<float,3>>);

    m.def("translateAntsImageF2_linear",  &translateAntsImage<itk::Image<float,2>, itk::LinearInterpolateImageFunction<itk::Image<float,2>, float>, float, 2>);
    m.def("translateAntsImageF2_nearest", &translateAntsImage<itk::Image<float,2>, itk::NearestNeighborInterpolateImageFunction<itk::Image<float,2>, float>, float, 2>);
    m.def("translateAntsImageF3_linear",  &translateAntsImage<itk::Image<float,3>, itk::LinearInterpolateImageFunction<itk::Image<float,3>, float>, float, 3>);
    m.def("translateAntsImageF3_nearest", &translateAntsImage<itk::Image<float,3>, itk::NearestNeighborInterpolateImageFunction<itk::Image<float,3>, float>, float, 3>);

    m.def("scaleAntsImageF2_linear",  &scaleAntsImage<itk::Image<float,2>, itk::LinearInterpolateImageFunction<itk::Image<float,2>, float>, float, 2>);
    m.def("scaleAntsImageF2_nearest", &scaleAntsImage<itk::Image<float,2>, itk::NearestNeighborInterpolateImageFunction<itk::Image<float,2>, float>, float, 2>);
    m.def("scaleAntsImageF3_linear",  &scaleAntsImage<itk::Image<float,3>, itk::LinearInterpolateImageFunction<itk::Image<float,3>, float>, float, 3>);
    m.def("scaleAntsImageF3_nearest", &scaleAntsImage<itk::Image<float,3>, itk::NearestNeighborInterpolateImageFunction<itk::Image<float,3>, float>, float, 3>);

}






