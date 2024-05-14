#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <tuple>
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itkPyBuffer.h"

#include "readImage.h"
#include "antsImage.h"


namespace nb = nanobind;
using namespace nb::literals;


template <typename ImageType>
AntsImage<ImageType> imageRead( std::string filename )
{
    typedef typename ImageType::Pointer           ImagePointerType;
    typedef itk::ImageFileReader< ImageType >     ImageReaderType;

    typename ImageReaderType::Pointer image_reader = ImageReaderType::New() ;
    image_reader->SetFileName( filename.c_str() ) ;
    image_reader->Update();

    ImagePointerType itkImage = image_reader->GetOutput();
    AntsImage<ImageType> myImage = { itkImage };
    return myImage;
}


template <typename ImageType>
AntsImage<ImageType> fromNumpy( nb::ndarray<nb::numpy> data, nb::tuple datashape )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType antsImage = ImageType::New();
    typedef itk::PyBuffer<ImageType> PyBufferType;

    nb::object o = nb::cast(data);
    antsImage = PyBufferType::_GetImageViewFromArray(o.ptr(), datashape.ptr(), nb::make_tuple(1)[0].ptr());

    AntsImage<ImageType> myImage = { antsImage };
    return myImage;
}


void local_readImage(nb::module_ &m) {

    m.def("imageReadUC2", &imageRead<itk::Image<unsigned char,2>>);
    m.def("imageReadUC3", &imageRead<itk::Image<unsigned char,3>>);
    m.def("imageReadUC4", &imageRead<itk::Image<unsigned char,4>>);
    m.def("imageReadUI2", &imageRead<itk::Image<unsigned int,2>>);
    m.def("imageReadUI3", &imageRead<itk::Image<unsigned int,3>>);
    m.def("imageReadUI4", &imageRead<itk::Image<unsigned int,4>>);
    m.def("imageReadF2", &imageRead<itk::Image<float,2>>);
    m.def("imageReadF3", &imageRead<itk::Image<float,3>>);
    m.def("imageReadF4", &imageRead<itk::Image<float,4>>);
    m.def("imageReadD2", &imageRead<itk::Image<double,2>>);
    m.def("imageReadD3", &imageRead<itk::Image<double,3>>);
    m.def("imageReadD4", &imageRead<itk::Image<double,4>>);
    m.def("imageReadVUC2", &imageRead<itk::VectorImage<unsigned char,2>>);
    m.def("imageReadVUC3", &imageRead<itk::VectorImage<unsigned char,3>>);
    m.def("imageReadVUC4", &imageRead<itk::VectorImage<unsigned char,4>>);
    m.def("imageReadVUI2", &imageRead<itk::VectorImage<unsigned int,2>>);
    m.def("imageReadVUI3", &imageRead<itk::VectorImage<unsigned int,3>>);
    m.def("imageReadVUI4", &imageRead<itk::VectorImage<unsigned int,4>>);
    m.def("imageReadVF2", &imageRead<itk::VectorImage<float,2>>);
    m.def("imageReadVF3", &imageRead<itk::VectorImage<float,3>>);
    m.def("imageReadVF4", &imageRead<itk::VectorImage<float,4>>);
    m.def("imageReadVD2", &imageRead<itk::VectorImage<double,2>>);
    m.def("imageReadVD3", &imageRead<itk::VectorImage<double,3>>);
    m.def("imageReadVD4", &imageRead<itk::VectorImage<double,4>>);
    m.def("imageReadRGBUC2", &imageRead<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("imageReadRGBUC3", &imageRead<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("imageReadRGBF2", &imageRead<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("imageReadRGBF3", &imageRead<itk::Image<itk::RGBPixel<float>,3>>);


    m.def("fromNumpyUC2", &fromNumpy<itk::Image<unsigned char,2>>);
    m.def("fromNumpyUC3", &fromNumpy<itk::Image<unsigned char,3>>);
    m.def("fromNumpyUC4", &fromNumpy<itk::Image<unsigned char,4>>);
    m.def("fromNumpyUI2", &fromNumpy<itk::Image<unsigned int,2>>);
    m.def("fromNumpyUI3", &fromNumpy<itk::Image<unsigned int,3>>);
    m.def("fromNumpyUI4", &fromNumpy<itk::Image<unsigned int,4>>);
    m.def("fromNumpyF2", &fromNumpy<itk::Image<float,2>>);
    m.def("fromNumpyF3", &fromNumpy<itk::Image<float,3>>);
    m.def("fromNumpyF4", &fromNumpy<itk::Image<float,4>>);
    m.def("fromNumpyD2", &fromNumpy<itk::Image<double,2>>);
    m.def("fromNumpyD3", &fromNumpy<itk::Image<double,3>>);
    m.def("fromNumpyD4", &fromNumpy<itk::Image<double,4>>);
}