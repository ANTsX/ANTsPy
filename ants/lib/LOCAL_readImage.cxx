#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <tuple>
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itkPyBuffer.h"

#include "LOCAL_readImage.h"
#include "LOCAL_antsImage.h"


namespace py = pybind11;


template< typename ImageType >
py::capsule imageReadHelper( std::string filename )
{
    //py::print("at imagereadhelper");
    typedef typename ImageType::Pointer           ImagePointerType;
    typedef itk::ImageFileReader< ImageType >     ImageReaderType;

    typename ImageReaderType::Pointer image_reader = ImageReaderType::New() ;
    image_reader->SetFileName( filename.c_str() ) ;
    image_reader->Update();

    ImagePointerType itkImage = image_reader->GetOutput();
    return wrap<ImageType>( itkImage );
}

template <typename ImageType>
py::capsule imageRead( std::string filename )
{
    py::capsule antsImage;
    antsImage = imageReadHelper<ImageType>( filename );
    return antsImage;
}


template <typename ImageType>
py::capsule fromNumpy( py::array data, py::tuple datashape )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType myimage = ImageType::New();
    typedef itk::PyBuffer<ImageType> PyBufferType;

    //py::tuple datashape = py::getattr(data, "shape");
    myimage = PyBufferType::_GetImageViewFromArray( data.ptr(), datashape.ptr(), py::make_tuple(1)[0].ptr() );

    return wrap<ImageType>( myimage );
}


void testPrint( py::object &myImage )
{
    unsigned int ndim = py::getattr(myImage, "dimension").cast<unsigned int>();
    std::string pixeltype = py::getattr(myImage, "pixeltype").cast<std::string>();
}



template <typename ImageType>
void wrapReadImage(py::module & m, std::string const & suffix) {
    m.def(("imageRead" + suffix).c_str(), &imageRead<ImageType>,
         "Read ANTsImage from file");
}



PYBIND11_MODULE(readImage, m) {

    wrapReadImage<itk::Image<unsigned char, 2>>(m, "UC2");
    wrapReadImage<itk::Image<unsigned char, 3>>(m, "UC3");
    wrapReadImage<itk::Image<unsigned char, 4>>(m, "UC4");
    wrapReadImage<itk::Image<unsigned int, 2>>(m, "UI2");
    wrapReadImage<itk::Image<unsigned int, 3>>(m, "UI3");
    wrapReadImage<itk::Image<unsigned int, 4>>(m, "UI4");
    wrapReadImage<itk::Image<float, 2>>(m, "F2");
    wrapReadImage<itk::Image<float, 3>>(m, "F3");
    wrapReadImage<itk::Image<float, 4>>(m, "F4");
    wrapReadImage<itk::Image<double, 2>>(m, "D2");
    wrapReadImage<itk::Image<double, 3>>(m, "D3");
    wrapReadImage<itk::Image<double, 4>>(m, "D4");

    wrapReadImage<itk::VectorImage<unsigned char, 2>>(m, "VUC2");
    wrapReadImage<itk::VectorImage<unsigned char, 3>>(m, "VUC3");
    wrapReadImage<itk::VectorImage<unsigned char, 4>>(m, "VUC4");
    wrapReadImage<itk::VectorImage<unsigned int, 2>>(m, "VUI2");
    wrapReadImage<itk::VectorImage<unsigned int, 3>>(m, "VUI3");
    wrapReadImage<itk::VectorImage<unsigned int, 4>>(m, "VUI4");
    wrapReadImage<itk::VectorImage<float, 2>>(m, "VF2");
    wrapReadImage<itk::VectorImage<float, 3>>(m, "VF3");
    wrapReadImage<itk::VectorImage<float, 4>>(m, "VF4");
    wrapReadImage<itk::VectorImage<double, 2>>(m, "VD2");
    wrapReadImage<itk::VectorImage<double, 3>>(m, "VD3");
    wrapReadImage<itk::VectorImage<double, 4>>(m, "VD4");

    wrapReadImage<itk::Image<itk::RGBPixel<unsigned char>, 2>>(m, "RGBUC2");
    wrapReadImage<itk::Image<itk::RGBPixel<unsigned char>, 3>>(m, "RGBUC3");

    m.def("fromNumpyUC2", &fromNumpy<itk::Image<unsigned char, 2>>);
    m.def("fromNumpyUC3", &fromNumpy<itk::Image<unsigned char, 3>>);
    m.def("fromNumpyUC4", &fromNumpy<itk::Image<unsigned char, 4>>);
    m.def("fromNumpyUI2", &fromNumpy<itk::Image<unsigned int, 2>>);
    m.def("fromNumpyUI3", &fromNumpy<itk::Image<unsigned int, 3>>);
    m.def("fromNumpyUI4", &fromNumpy<itk::Image<unsigned int, 4>>);
    m.def("fromNumpyF2", &fromNumpy<itk::Image<float, 2>>);
    m.def("fromNumpyF3", &fromNumpy<itk::Image<float, 3>>);
    m.def("fromNumpyF4", &fromNumpy<itk::Image<float, 4>>);
    m.def("fromNumpyD2", &fromNumpy<itk::Image<double, 2>>);
    m.def("fromNumpyD3", &fromNumpy<itk::Image<double, 3>>);
    m.def("fromNumpyD4", &fromNumpy<itk::Image<double, 4>>);
    //m.def("fromNumpyRGBUC2", &fromNumpy<itk::Image<itk::RGBPixel<unsigned char>, 2>>);
    //m.def("fromNumpyRGBUC3", &fromNumpy<itk::Image<itk::RGBPixel<unsigned char>, 3>>);

}


