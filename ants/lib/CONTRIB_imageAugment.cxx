
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkNormalizeImageFilter.h"
#include "itkSigmoidImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkTranslationTransform.h"

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
    CastFilterType::Pointer castFilter = CastFilterType::New();
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
    RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
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
    ShiftScaleFilterType::Pointer shiftFilter = ShiftScaleFilterType::New();
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
    NormalizeFilterType::Pointer normalizeFilter = NormalizeFilterType::New();
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
    SigmoidFilterType::Pointer sigmoidFilter = SigmoidFilterType::New();
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

    typedef itk::FlipImageFilter< ImageType, ImageType > FlipFilterType;
    FlipFilterType::Pointer flipFilter = FlipFilterType::New();
    flipFilter->SetInput( itkImage );

    typedef FlipFilterType::FlipAxesArrayType FlipAxesArrayType;
    FlipAxesArrayType flipArray;
    flipArray[0] = axis1;
    flipArray[1] = axis2;
    filter->SetFlipAxes( flipArray );

    flipFilter->Update();
    return wrap< ImageType >( flipFilter->GetOutput() );
}

/*
Apply a translation to an ANTsImage. This function uses itk::TranslationTransform
which is faster than using itk::AffineTransform.
*/
template <typename ImageType>
py::capsule translateAntsImage( py::capsule & antsImage, py::capsule referenceImage, std::vector<int> translationList )
{
    // first: unwrap ANTsImage
    typename ImageType::Pointer itkImage = as< ImageType >( antsImage );

    typedef itk::TranslationTransform<float,2> TranslationTransformType;
    TranslationTransformType::Pointer transform = TranslationTransformType::New();
    
    TranslationTransformType::OutputVectorType translation;
    for (unsigned int i; i < translationList.size(); ++i )
    {
        translation[i] = translationList[i]
    }
    transform->Translate(translation);

    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
    ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetTransform(transform.GetPointer());
    resampleFilter->SetInput(image);

    ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
    resampleFilter->SetSize( size );

}


PYBIND11_MODULE(imageAugmentModule, m)
{
    m.def("castAntsImageUC2F2", &castAntsImage<itk::Image<unsigned char,2>,itk::Image<float,2>>);
    m.def("castAntsImageUI2F2", &castAntsImage<itk::Image<unsigned int, 2>,itk::Image<float,2>>);
    m.def("castAntsImageD2F2", &castAntsImage<itk::Image<double,2>,itk::Image<float,2>>);
    m.def("castAntsImageUC3F3", &castAntsImage<itk::Image<unsigned char,3>,itk::Image<float,3>>);
    m.def("castAntsImageUI3F3", &castAntsImage<itk::Image<unsigned int, 3>,itk::Image<float,3>>);
    m.def("castAntsImageD3F3", &castAntsImage<itk::Image<double,3>,itk::Image<float,3>>);
    m.def("castAntsImageUC4F4", &castAntsImage<itk::Image<unsigned char,4>,itk::Image<float,4>>);
    m.def("castAntsImageUI4F4", &castAntsImage<itk::Image<unsigned int, 4>,itk::Image<float,4>>);
    m.def("castAntsImageD4F4", &castAntsImage<itk::Image<double,4>,itk::Image<float,4>>);

    m.def("rescaleAntsImageF2", &rescaleAntsImage<itk::Image<float,2>>);
    m.def("rescaleAntsImageF3", &rescaleAntsImage<itk::Image<float,3>>);
    m.def("rescaleAntsImageF4", &rescaleAntsImage<itk::Image<float,4>>);

    m.def("shiftScaleAntsImageF2", &shiftScaleAntsImage<itk::Image<float,2>>);
    m.def("shiftScaleAntsImageF3", &shiftScaleAntsImage<itk::Image<float,3>>);
    m.def("shiftScaleAntsImageF4", &shiftScaleAntsImage<itk::Image<float,4>>);

    m.def("normalizeAntsImageF2", &normalizeAntsImage<itk::Image<float,2>>);
    m.def("normalizeAntsImageF3", &normalizeAntsImage<itk::Image<float,3>>);
    m.def("normalizeAntsImageF4", &normalizeAntsImage<itk::Image<float,4>>);

    m.def("sigmoidTransformAntsImageF2", &sigmoidTransformAntsImage<itk::Image<float,2>>);
    m.def("sigmoidTransformAntsImageF3", &sigmoidTransformAntsImage<itk::Image<float,3>>);
    m.def("sigmoidTransformAntsImageF4", &sigmoidTransformAntsImage<itk::Image<float,4>>);

    m.def("flipAntsImageF2", &flipAntsImage<itk::Image<float,2>>);
    m.def("flipAntsImageF3", &flipAntsImage<itk::Image<float,3>>);
    m.def("flipAntsImageF4", &flipAntsImage<itk::Image<float,4>>);
}






