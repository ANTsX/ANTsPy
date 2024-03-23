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

template< typename ImageType >
void * imageReadHelper( std::string filename )
{
    typedef typename ImageType::Pointer           ImagePointerType;
    typedef itk::ImageFileReader< ImageType >     ImageReaderType;

    typename ImageReaderType::Pointer image_reader = ImageReaderType::New() ;
    image_reader->SetFileName( filename.c_str() ) ;
    image_reader->Update();

    ImagePointerType itkImage = image_reader->GetOutput();
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType * ptr = new ImagePointerType( itkImage );
    return ptr;
}

void * imageRead( std::string filename, std::string imageType )
{

    if (imageType == "UC3") {
        void * antsImage;
        using ImageType = itk::Image<unsigned char, 3>;
        antsImage = imageReadHelper<ImageType>(filename);
        return antsImage;
    }

    if (imageType == "UI3") {
        void * antsImage;
        using ImageType = itk::Image<unsigned int, 3>;
        antsImage = imageReadHelper<ImageType>(filename);
        return antsImage;
    }

    if (imageType == "F3") {
        void * antsImage;
        using ImageType = itk::Image<float, 3>;
        antsImage = imageReadHelper<ImageType>(filename);
        return antsImage;
    }

    if (imageType == "D3") {
        void * antsImage;
        using ImageType = itk::Image<double, 3>;
        antsImage = imageReadHelper<ImageType>(filename);
        return antsImage;
    }

}

void local_imageRead(nb::module_ &m) {
    m.def("imageRead", &imageRead);
}


