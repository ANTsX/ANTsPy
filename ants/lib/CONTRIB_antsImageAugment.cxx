
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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


#include "itkImage.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

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
py::capsule sigmoidTransformAntsImage( py::capsule & antsImage, 
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
py::capsule translateAntsImage( py::capsule & inputAntsImage, py::capsule refAntsImage, 
                                std::vector<float> translationList, std::string interpolationType )
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


PYBIND11_MODULE(antsImageAugment, m)
{
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

    m.def("sigmoidTransformAntsImageF2", &sigmoidTransformAntsImage<itk::Image<float,2>>);
    m.def("sigmoidTransformAntsImageF3", &sigmoidTransformAntsImage<itk::Image<float,3>>);

    m.def("flipAntsImageF2", &flipAntsImage<itk::Image<float,2>>);
    m.def("flipAntsImageF3", &flipAntsImage<itk::Image<float,3>>);

    m.def("translateAntsImageF2_linear", &translateAntsImage<itk::Image<float,2>, itk::LinearInterpolateImageFunction<itk::Image<float,2>, float>, float, 2>);
    m.def("translateAntsImageF2_nearest", &translateAntsImage<itk::Image<float,2>, itk::NearestNeighborInterpolateImageFunction<itk::Image<float,2>, float>, float, 2>);
    m.def("translateAntsImageF3_linear", &translateAntsImage<itk::Image<float,3>, itk::LinearInterpolateImageFunction<itk::Image<float,3>, float>, float, 3>);
    m.def("translateAntsImageF3_nearest", &translateAntsImage<itk::Image<float,3>, itk::NearestNeighborInterpolateImageFunction<itk::Image<float,3>, float>, float, 3>);

}






