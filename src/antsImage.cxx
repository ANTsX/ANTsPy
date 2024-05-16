#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include "itkPyBuffer.h"
#include "itkImageIOBase.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include <iostream>

#include "antsImage.h"


namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType>
nb::object toNumpy( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typedef itk::PyBuffer<ImageType> PyBufferType;
    PyObject * itkArray = PyBufferType::_GetArrayViewFromImage( image );
    nb::object itkArrayObject = nb::steal( itkArray );
    return itkArrayObject;
}


std::string ptrstr2(void * c)
{
    std::stringstream ss;
    ss << (void const *)c;
    std::string s = ss.str();
    return s;
}


/*
The return value (const char * file) from this function should be able to go
through the following code block without segfaulting. This is what ANTs uses
to convert back to an itk image.

void *             ptr;
sscanf(file, "%p", (void **)&ptr);
typename ImageType::Pointer newImage = *(static_cast<typename ImageType::Pointer *>(ptr));
*/
template <typename ImageType>
std::string ptrstr(AntsImage<ImageType> & myPointer)
{
    typename ImageType::Pointer * itkImage = & myPointer.ptr;
    std::stringstream ss;
    ss << (void const *)itkImage;
    std::string s = ss.str();
    return s;
}

void local_antsImage(nb::module_ &m) {

    m.def("ptrstr",  &ptrstr<itk::Image<unsigned char,2>>);
    m.def("ptrstr",  &ptrstr<itk::Image<unsigned char,3>>);
    m.def("ptrstr",  &ptrstr<itk::Image<unsigned char,4>>);
    m.def("ptrstr",  &ptrstr<itk::Image<unsigned int,2>>);
    m.def("ptrstr",  &ptrstr<itk::Image<unsigned int,3>>);
    m.def("ptrstr",  &ptrstr<itk::Image<unsigned int,4>>);
    m.def("ptrstr",   &ptrstr<itk::Image<float,2>>);
    m.def("ptrstr",   &ptrstr<itk::Image<float,3>>);
    m.def("ptrstr",   &ptrstr<itk::Image<float,4>>);
    m.def("ptrstr",   &ptrstr<itk::Image<double,2>>);
    m.def("ptrstr",   &ptrstr<itk::Image<double,3>>);
    m.def("ptrstr",   &ptrstr<itk::Image<double,4>>);
    m.def("ptrstr", &ptrstr<itk::VectorImage<unsigned char,2>>);
    m.def("ptrstr", &ptrstr<itk::VectorImage<unsigned char,3>>);
    m.def("ptrstr", &ptrstr<itk::VectorImage<unsigned char,4>>);
    m.def("ptrstr", &ptrstr<itk::VectorImage<unsigned int,2>>);
    m.def("ptrstr", &ptrstr<itk::VectorImage<unsigned int,3>>);
    m.def("ptrstr", &ptrstr<itk::VectorImage<unsigned int,4>>);
    m.def("ptrstr",  &ptrstr<itk::VectorImage<float,2>>);
    m.def("ptrstr",  &ptrstr<itk::VectorImage<float,3>>);
    m.def("ptrstr",  &ptrstr<itk::VectorImage<float,4>>);
    m.def("ptrstr",  &ptrstr<itk::VectorImage<double,2>>);
    m.def("ptrstr",  &ptrstr<itk::VectorImage<double,3>>);
    m.def("ptrstr",  &ptrstr<itk::VectorImage<double,4>>);
    m.def("ptrstr", &ptrstr<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("ptrstr", &ptrstr<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("ptrstr", &ptrstr<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("ptrstr", &ptrstr<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getComponents", &getComponents<itk::VectorImage<unsigned char,2>>);
    m.def("getComponents", &getComponents<itk::VectorImage<unsigned char,3>>);
    m.def("getComponents", &getComponents<itk::VectorImage<unsigned char,4>>);
    m.def("getComponents", &getComponents<itk::VectorImage<unsigned int,2>>);
    m.def("getComponents", &getComponents<itk::VectorImage<unsigned int,3>>);
    m.def("getComponents", &getComponents<itk::VectorImage<unsigned int,4>>);
    m.def("getComponents",  &getComponents<itk::VectorImage<float,2>>);
    m.def("getComponents",  &getComponents<itk::VectorImage<float,3>>);
    m.def("getComponents",  &getComponents<itk::VectorImage<float,4>>);
    m.def("getComponents",  &getComponents<itk::VectorImage<double,2>>);
    m.def("getComponents",  &getComponents<itk::VectorImage<double,3>>);
    m.def("getComponents",  &getComponents<itk::VectorImage<double,4>>);

    m.def("getShape",  &getShape<itk::Image<unsigned char,2>>);
    m.def("getShape",  &getShape<itk::Image<unsigned char,3>>);
    m.def("getShape",  &getShape<itk::Image<unsigned char,4>>);
    m.def("getShape",  &getShape<itk::Image<unsigned int,2>>);
    m.def("getShape",  &getShape<itk::Image<unsigned int,3>>);
    m.def("getShape",  &getShape<itk::Image<unsigned int,4>>);
    m.def("getShape",   &getShape<itk::Image<float,2>>);
    m.def("getShape",   &getShape<itk::Image<float,3>>);
    m.def("getShape",   &getShape<itk::Image<float,4>>);
    m.def("getShape",   &getShape<itk::Image<double,2>>);
    m.def("getShape",   &getShape<itk::Image<double,3>>);
    m.def("getShape",   &getShape<itk::Image<double,4>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned char,2>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned char,3>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned char,4>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned int,2>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned int,3>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned int,4>>);
    m.def("getShape",  &getShape<itk::VectorImage<float,2>>);
    m.def("getShape",  &getShape<itk::VectorImage<float,3>>);
    m.def("getShape",  &getShape<itk::VectorImage<float,4>>);
    m.def("getShape",  &getShape<itk::VectorImage<double,2>>);
    m.def("getShape",  &getShape<itk::VectorImage<double,3>>);
    m.def("getShape",  &getShape<itk::VectorImage<double,4>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getOrigin",  &getOrigin<itk::Image<unsigned char,2>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned char,3>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned char,4>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned int,2>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned int,3>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned int,4>>);
    m.def("getOrigin",   &getOrigin<itk::Image<float,2>>);
    m.def("getOrigin",   &getOrigin<itk::Image<float,3>>);
    m.def("getOrigin",   &getOrigin<itk::Image<float,4>>);
    m.def("getOrigin",   &getOrigin<itk::Image<double,2>>);
    m.def("getOrigin",   &getOrigin<itk::Image<double,3>>);
    m.def("getOrigin",   &getOrigin<itk::Image<double,4>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned char,2>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned char,3>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned char,4>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned int,2>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned int,3>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned int,4>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<float,2>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<float,3>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<float,4>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<double,2>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<double,3>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<double,4>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setOrigin",  &setOrigin<itk::Image<unsigned char,2>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned char,3>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned char,4>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned int,2>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned int,3>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned int,4>>);
    m.def("setOrigin",   &setOrigin<itk::Image<float,2>>);
    m.def("setOrigin",   &setOrigin<itk::Image<float,3>>);
    m.def("setOrigin",   &setOrigin<itk::Image<float,4>>);
    m.def("setOrigin",   &setOrigin<itk::Image<double,2>>);
    m.def("setOrigin",   &setOrigin<itk::Image<double,3>>);
    m.def("setOrigin",   &setOrigin<itk::Image<double,4>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned char,2>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned char,3>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned char,4>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned int,2>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned int,3>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned int,4>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<float,2>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<float,3>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<float,4>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<double,2>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<double,3>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<double,4>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getSpacing",  &getSpacing<itk::Image<unsigned char,2>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned char,3>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned char,4>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned int,2>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned int,3>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned int,4>>);
    m.def("getSpacing",   &getSpacing<itk::Image<float,2>>);
    m.def("getSpacing",   &getSpacing<itk::Image<float,3>>);
    m.def("getSpacing",   &getSpacing<itk::Image<float,4>>);
    m.def("getSpacing",   &getSpacing<itk::Image<double,2>>);
    m.def("getSpacing",   &getSpacing<itk::Image<double,3>>);
    m.def("getSpacing",   &getSpacing<itk::Image<double,4>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned char,2>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned char,3>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned char,4>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned int,2>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned int,3>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned int,4>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<float,2>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<float,3>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<float,4>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<double,2>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<double,3>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<double,4>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setSpacing",  &setSpacing<itk::Image<unsigned char,2>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned char,3>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned char,4>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned int,2>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned int,3>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned int,4>>);
    m.def("setSpacing",   &setSpacing<itk::Image<float,2>>);
    m.def("setSpacing",   &setSpacing<itk::Image<float,3>>);
    m.def("setSpacing",   &setSpacing<itk::Image<float,4>>);
    m.def("setSpacing",   &setSpacing<itk::Image<double,2>>);
    m.def("setSpacing",   &setSpacing<itk::Image<double,3>>);
    m.def("setSpacing",   &setSpacing<itk::Image<double,4>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned char,2>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned char,3>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned char,4>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned int,2>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned int,3>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned int,4>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<float,2>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<float,3>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<float,4>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<double,2>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<double,3>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<double,4>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getDirection",  &getDirection<itk::Image<unsigned char,2>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned char,3>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned char,4>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned int,2>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned int,3>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned int,4>>);
    m.def("getDirection",   &getDirection<itk::Image<float,2>>);
    m.def("getDirection",   &getDirection<itk::Image<float,3>>);
    m.def("getDirection",   &getDirection<itk::Image<float,4>>);
    m.def("getDirection",   &getDirection<itk::Image<double,2>>);
    m.def("getDirection",   &getDirection<itk::Image<double,3>>);
    m.def("getDirection",   &getDirection<itk::Image<double,4>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned char,2>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned char,3>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned char,4>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned int,2>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned int,3>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned int,4>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<float,2>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<float,3>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<float,4>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<double,2>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<double,3>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<double,4>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setDirection",  &setDirection<itk::Image<unsigned char,2>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned char,3>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned char,4>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned int,2>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned int,3>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned int,4>>);
    m.def("setDirection",   &setDirection<itk::Image<float,2>>);
    m.def("setDirection",   &setDirection<itk::Image<float,3>>);
    m.def("setDirection",   &setDirection<itk::Image<float,4>>);
    m.def("setDirection",   &setDirection<itk::Image<double,2>>);
    m.def("setDirection",   &setDirection<itk::Image<double,3>>);
    m.def("setDirection",   &setDirection<itk::Image<double,4>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned char,2>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned char,3>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned char,4>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned int,2>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned int,3>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned int,4>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<float,2>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<float,3>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<float,4>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<double,2>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<double,3>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<double,4>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("toFile",  &toFile<itk::Image<unsigned char,2>>);
    m.def("toFile",  &toFile<itk::Image<unsigned char,3>>);
    m.def("toFile",  &toFile<itk::Image<unsigned char,4>>);
    m.def("toFile",  &toFile<itk::Image<unsigned int,2>>);
    m.def("toFile",  &toFile<itk::Image<unsigned int,3>>);
    m.def("toFile",  &toFile<itk::Image<unsigned int,4>>);
    m.def("toFile",   &toFile<itk::Image<float,2>>);
    m.def("toFile",   &toFile<itk::Image<float,3>>);
    m.def("toFile",   &toFile<itk::Image<float,4>>);
    m.def("toFile",   &toFile<itk::Image<double,2>>);
    m.def("toFile",   &toFile<itk::Image<double,3>>);
    m.def("toFile",   &toFile<itk::Image<double,4>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned char,2>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned char,3>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned char,4>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned int,2>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned int,3>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned int,4>>);
    m.def("toFile",  &toFile<itk::VectorImage<float,2>>);
    m.def("toFile",  &toFile<itk::VectorImage<float,3>>);
    m.def("toFile",  &toFile<itk::VectorImage<float,4>>);
    m.def("toFile",  &toFile<itk::VectorImage<double,2>>);
    m.def("toFile",  &toFile<itk::VectorImage<double,3>>);
    m.def("toFile",  &toFile<itk::VectorImage<double,4>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("toNumpy",  &toNumpy<itk::Image<unsigned char,2>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned char,3>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned char,4>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned int,2>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned int,3>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned int,4>>);
    m.def("toNumpy",   &toNumpy<itk::Image<float,2>>);
    m.def("toNumpy",   &toNumpy<itk::Image<float,3>>);
    m.def("toNumpy",   &toNumpy<itk::Image<float,4>>);
    m.def("toNumpy",   &toNumpy<itk::Image<double,2>>);
    m.def("toNumpy",   &toNumpy<itk::Image<double,3>>);
    m.def("toNumpy",   &toNumpy<itk::Image<double,4>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned char,2>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned char,3>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned char,4>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned int,2>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned int,3>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned int,4>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<float,2>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<float,3>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<float,4>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<double,2>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<double,3>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<double,4>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<float>,3>>);

    nb::class_<AntsImage<itk::Image<unsigned char,2>>>(m, "AntsImageUC2");
    nb::class_<AntsImage<itk::Image<unsigned char,3>>>(m, "AntsImageUC3");
    nb::class_<AntsImage<itk::Image<unsigned char,4>>>(m, "AntsImageUC4");
    nb::class_<AntsImage<itk::Image<unsigned int,2>>>(m, "AntsImageUI2");
    nb::class_<AntsImage<itk::Image<unsigned int,3>>>(m, "AntsImageUI3");
    nb::class_<AntsImage<itk::Image<unsigned int,4>>>(m, "AntsImageUI4");
    nb::class_<AntsImage<itk::Image<float,2>>>(m, "AntsImageF2");
    nb::class_<AntsImage<itk::Image<float,3>>>(m, "AntsImageF3");
    nb::class_<AntsImage<itk::Image<float,4>>>(m, "AntsImageF4");
    nb::class_<AntsImage<itk::Image<double,2>>>(m, "AntsImageD2");
    nb::class_<AntsImage<itk::Image<double,3>>>(m, "AntsImageD3");
    nb::class_<AntsImage<itk::Image<double,4>>>(m, "AntsImageD4");
    nb::class_<AntsImage<itk::VectorImage<unsigned char,2>>>(m, "AntsImageVUC2");
    nb::class_<AntsImage<itk::VectorImage<unsigned char,3>>>(m, "AntsImageVUC3");
    nb::class_<AntsImage<itk::VectorImage<unsigned char,4>>>(m, "AntsImageVUC4");
    nb::class_<AntsImage<itk::VectorImage<unsigned int,2>>>(m, "AntsImageVUI2");
    nb::class_<AntsImage<itk::VectorImage<unsigned int,3>>>(m, "AntsImageVUI3");
    nb::class_<AntsImage<itk::VectorImage<unsigned int,4>>>(m, "AntsImageVUI4");
    nb::class_<AntsImage<itk::VectorImage<float,2>>>(m, "AntsImageVF2");
    nb::class_<AntsImage<itk::VectorImage<float,3>>>(m, "AntsImageVF3");
    nb::class_<AntsImage<itk::VectorImage<float,4>>>(m, "AntsImageVF4");
    nb::class_<AntsImage<itk::VectorImage<double,2>>>(m, "AntsImageVD2");
    nb::class_<AntsImage<itk::VectorImage<double,3>>>(m, "AntsImageVD3");
    nb::class_<AntsImage<itk::VectorImage<double,4>>>(m, "AntsImageVD4");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<unsigned char>,2>>>(m, "AntsImageRGBUC2");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<unsigned char>,3>>>(m, "AntsImageRGBUC3");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<float>,2>>>(m, "AntsImageRGBF2");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<float>,3>>>(m, "AntsImageRGBF3");
}
