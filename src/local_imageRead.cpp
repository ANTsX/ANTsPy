#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <tuple>
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"


namespace nb = nanobind;

using namespace nb::literals;

template <typename ImageType>
typename ImageType::Pointer imageReadHelper( std::string filename )
{
    typedef typename ImageType::Pointer           ImagePointerType;
    typedef itk::ImageFileReader< ImageType >     ImageReaderType;

    typename ImageReaderType::Pointer image_reader = ImageReaderType::New() ;
    image_reader->SetFileName( filename.c_str() ) ;
    image_reader->Update();

    ImagePointerType itkImage = image_reader->GetOutput();
    return itkImage;
}


template <typename ImageType>
AntsImage<ImageType> imageRead( std::string filename, std::string imageType )
{

    typename ImageType::Pointer  antsImage = imageReadHelper<ImageType>(filename);

    AntsImage<ImageType> myImage = { antsImage };
    return myImage;
}



void local_imageRead(nb::module_ &m) {
    m.def("imageReadF3", &imageRead<itk::Image<float, 3>>);
    m.def("imageReadF2", &imageRead<itk::Image<float, 2>>);

    nb::class_<AntsImage<itk::Image<float, 3>>>(m, "AntsImageF3");
    nb::class_<AntsImage<itk::Image<float, 2>>>(m, "AntsImageF2");
}


