// NEED THESE INCLUDES FOR PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "itkImage.h"
#include "itkScaleTransform.h"
#include "itkResampleImageFilter.h"

// NEED THIS INCLUDE FOR WRAPPING/UNWRAPPING
#include "../LOCAL_antsImage.h"

// NEED THIS FOR PYBIND11
namespace py = pybind11;

template <typename ImageType>
py::capsule scaleImage2D( py::capsule myAntsImage, float scale1, float scale2 )
{  
    // UN-WRAP THE CAPSULE
    typename ImageType::Pointer myImage = as<ImageType>( myAntsImage );

    typedef itk::ScaleTransform<double, 2> TransformType;
    typename TransformType::Pointer scaleTransform = TransformType::New();
    itk::FixedArray<float, 2> scale;
    scale[0] = scale1; // newWidth/oldWidth
    scale[1] = scale2;
    scaleTransform->SetScale( scale );

    itk::Point<float,2> center;
    center[0] = myImage->GetLargestPossibleRegion().GetSize()[0]/2;
    center[1] = myImage->GetLargestPossibleRegion().GetSize()[1]/2;

    scaleTransform->SetCenter( center );

    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
    typename ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetTransform( scaleTransform );
    resampleFilter->SetInput( myImage );
    resampleFilter->SetSize( myImage->GetLargestPossibleRegion().GetSize() );
    resampleFilter->Update();

    typename ImageType::Pointer resampledImage = resampleFilter->GetOutput();

    // WRAP THE ITK IMAGE 
    py::capsule resampledCapsule = wrap<ImageType>( resampledImage );
    return resampledCapsule;
}

// DECLARE OUR FUNCTION FOR APPROPRIATE TYPES 
PYBIND11_MODULE(scaleImageModule, m)
{
    m.def("scaleImage2D", &scaleImage2D<itk::Image<float,2>>);
}