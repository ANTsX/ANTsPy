
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkPyBuffer.h"

#include "itkMath.h"
#include "itkPyVnl.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix_fixed.hxx"
#include "vnl/vnl_transpose.h"
#include "vnl/algo/vnl_matrix_inverse.h"
#include "vnl/vnl_matrix.h"
#include "vnl/algo/vnl_determinant.h"
#include "LOCAL_antsImage.h"


namespace py = pybind11;

template <typename ImageType>
py::array numpyHelper( typename ImageType::Pointer itkImage )
{
    typedef itk::PyBuffer<ImageType> PyBufferType;
    PyObject * itkArray = PyBufferType::_GetArrayViewFromImage( itkImage );
    py::array itkArrayObject = py::reinterpret_steal<py::object>( itkArray );

    return itkArrayObject;

}

template <typename ImageType>
py::array toNumpy( py::capsule myPointer )
{
    typename ImageType::Pointer itkImage = as<ImageType>( myPointer );
    return numpyHelper<ImageType>( itkImage );
}

std::string ptrstr(py::capsule c)
{
    std::stringstream ss;
    ss << (void const *)c;
    std::string s = ss.str();
    return s;
}




PYBIND11_MODULE(antsImage, m) {
    m.def("ptrstr", &ptrstr);

    m.def(("getShapeUC2").c_str(),  &getShape<itk::Image<unsigned char,2>>);
    m.def(("getShapeUC3").c_str(),  &getShape<itk::Image<unsigned char,3>>);
    m.def(("getShapeUC4").c_str(),  &getShape<itk::Image<unsigned char,4>>);
    m.def(("getShapeUI2").c_str(),  &getShape<itk::Image<unsigned int,2>>);
    m.def(("getShapeUI3").c_str(),  &getShape<itk::Image<unsigned int,3>>);
    m.def(("getShapeUI4").c_str(),  &getShape<itk::Image<unsigned int,4>>);
    m.def(("getShapeF2").c_str(),   &getShape<itk::Image<float,2>>);
    m.def(("getShapeF3").c_str(),   &getShape<itk::Image<float,3>>);
    m.def(("getShapeF4").c_str(),   &getShape<itk::Image<float,4>>);
    m.def(("getShapeVUC2").c_str(), &getShape<itk::VectorImage<unsigned char,2>>);
    m.def(("getShapeVUC3").c_str(), &getShape<itk::VectorImage<unsigned char,3>>);
    m.def(("getShapeVUC4").c_str(), &getShape<itk::VectorImage<unsigned char,4>>);
    m.def(("getShapeVUI2").c_str(), &getShape<itk::VectorImage<unsigned int,2>>);
    m.def(("getShapeVUI3").c_str(), &getShape<itk::VectorImage<unsigned int,3>>);
    m.def(("getShapeVUI4").c_str(), &getShape<itk::VectorImage<unsigned int,4>>);
    m.def(("getShapeVF2").c_str(),  &getShape<itk::VectorImage<float,2>>);
    m.def(("getShapeVF3").c_str(),  &getShape<itk::VectorImage<float,3>>);
    m.def(("getShapeVF4").c_str(),  &getShape<itk::VectorImage<float,4>>);
    m.def(("getShapeRGBUC3").c_str(), &getShape<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("setShapeUC2").c_str(),  &setShape<itk::Image<unsigned char,2>>);
    m.def(("setShapeUC3").c_str(),  &setShape<itk::Image<unsigned char,3>>);
    m.def(("setShapeUC4").c_str(),  &setShape<itk::Image<unsigned char,4>>);
    m.def(("setShapeUI2").c_str(),  &setShape<itk::Image<unsigned int,2>>);
    m.def(("setShapeUI3").c_str(),  &setShape<itk::Image<unsigned int,3>>);
    m.def(("setShapeUI4").c_str(),  &setShape<itk::Image<unsigned int,4>>);
    m.def(("setShapeF2").c_str(),   &setShape<itk::Image<float,2>>);
    m.def(("setShapeF3").c_str(),   &setShape<itk::Image<float,3>>);
    m.def(("setShapeF4").c_str(),   &setShape<itk::Image<float,4>>);
    m.def(("setShapeVUC2").c_str(), &setShape<itk::VectorImage<unsigned char,2>>);
    m.def(("setShapeVUC3").c_str(), &setShape<itk::VectorImage<unsigned char,3>>);
    m.def(("setShapeVUC4").c_str(), &setShape<itk::VectorImage<unsigned char,4>>);
    m.def(("setShapeVUI2").c_str(), &setShape<itk::VectorImage<unsigned int,2>>);
    m.def(("setShapeVUI3").c_str(), &setShape<itk::VectorImage<unsigned int,3>>);
    m.def(("setShapeVUI4").c_str(), &setShape<itk::VectorImage<unsigned int,4>>);
    m.def(("setShapeVF2").c_str(),  &setShape<itk::VectorImage<float,2>>);
    m.def(("setShapeVF3").c_str(),  &setShape<itk::VectorImage<float,3>>);
    m.def(("setShapeVF4").c_str(),  &setShape<itk::VectorImage<float,4>>);
    m.def(("setShapeRGBUC3").c_str(), &setShape<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("getOriginUC2").c_str(),  &getOrigin<itk::Image<unsigned char,2>>);
    m.def(("getOriginUC3").c_str(),  &getOrigin<itk::Image<unsigned char,3>>);
    m.def(("getOriginUC4").c_str(),  &getOrigin<itk::Image<unsigned char,4>>);
    m.def(("getOriginUI2").c_str(),  &getOrigin<itk::Image<unsigned int,2>>);
    m.def(("getOriginUI3").c_str(),  &getOrigin<itk::Image<unsigned int,3>>);
    m.def(("getOriginUI4").c_str(),  &getOrigin<itk::Image<unsigned int,4>>);
    m.def(("getOriginF2").c_str(),   &getOrigin<itk::Image<float,2>>);
    m.def(("getOriginF3").c_str(),   &getOrigin<itk::Image<float,3>>);
    m.def(("getOriginF4").c_str(),   &getOrigin<itk::Image<float,4>>);
    m.def(("getOriginVUC2").c_str(), &getOrigin<itk::VectorImage<unsigned char,2>>);
    m.def(("getOriginVUC3").c_str(), &getOrigin<itk::VectorImage<unsigned char,3>>);
    m.def(("getOriginVUC4").c_str(), &getOrigin<itk::VectorImage<unsigned char,4>>);
    m.def(("getOriginVUI2").c_str(), &getOrigin<itk::VectorImage<unsigned int,2>>);
    m.def(("getOriginVUI3").c_str(), &getOrigin<itk::VectorImage<unsigned int,3>>);
    m.def(("getOriginVUI4").c_str(), &getOrigin<itk::VectorImage<unsigned int,4>>);
    m.def(("getOriginVF2").c_str(),  &getOrigin<itk::VectorImage<float,2>>);
    m.def(("getOriginVF3").c_str(),  &getOrigin<itk::VectorImage<float,3>>);
    m.def(("getOriginVF4").c_str(),  &getOrigin<itk::VectorImage<float,4>>);
    m.def(("getOriginRGBUC3").c_str(), &getOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("setOriginUC2").c_str(),  &setOrigin<itk::Image<unsigned char,2>>);
    m.def(("setOriginUC3").c_str(),  &setOrigin<itk::Image<unsigned char,3>>);
    m.def(("setOriginUC4").c_str(),  &setOrigin<itk::Image<unsigned char,4>>);
    m.def(("setOriginUI2").c_str(),  &setOrigin<itk::Image<unsigned int,2>>);
    m.def(("setOriginUI3").c_str(),  &setOrigin<itk::Image<unsigned int,3>>);
    m.def(("setOriginUI4").c_str(),  &setOrigin<itk::Image<unsigned int,4>>);
    m.def(("setOriginF2").c_str(),   &setOrigin<itk::Image<float,2>>);
    m.def(("setOriginF3").c_str(),   &setOrigin<itk::Image<float,3>>);
    m.def(("setOriginF4").c_str(),   &setOrigin<itk::Image<float,4>>);
    m.def(("setOriginVUC2").c_str(), &setOrigin<itk::VectorImage<unsigned char,2>>);
    m.def(("setOriginVUC3").c_str(), &setOrigin<itk::VectorImage<unsigned char,3>>);
    m.def(("setOriginVUC4").c_str(), &setOrigin<itk::VectorImage<unsigned char,4>>);
    m.def(("setOriginVUI2").c_str(), &setOrigin<itk::VectorImage<unsigned int,2>>);
    m.def(("setOriginVUI3").c_str(), &setOrigin<itk::VectorImage<unsigned int,3>>);
    m.def(("setOriginVUI4").c_str(), &setOrigin<itk::VectorImage<unsigned int,4>>);
    m.def(("setOriginVF2").c_str(),  &setOrigin<itk::VectorImage<float,2>>);
    m.def(("setOriginVF3").c_str(),  &setOrigin<itk::VectorImage<float,3>>);
    m.def(("setOriginVF4").c_str(),  &setOrigin<itk::VectorImage<float,4>>);
    m.def(("setOriginRGBUC3").c_str(), &setOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("getSpacingUC2").c_str(),  &getSpacing<itk::Image<unsigned char,2>>);
    m.def(("getSpacingUC3").c_str(),  &getSpacing<itk::Image<unsigned char,3>>);
    m.def(("getSpacingUC4").c_str(),  &getSpacing<itk::Image<unsigned char,4>>);
    m.def(("getSpacingUI2").c_str(),  &getSpacing<itk::Image<unsigned int,2>>);
    m.def(("getSpacingUI3").c_str(),  &getSpacing<itk::Image<unsigned int,3>>);
    m.def(("getSpacingUI4").c_str(),  &getSpacing<itk::Image<unsigned int,4>>);
    m.def(("getSpacingF2").c_str(),   &getSpacing<itk::Image<float,2>>);
    m.def(("getSpacingF3").c_str(),   &getSpacing<itk::Image<float,3>>);
    m.def(("getSpacingF4").c_str(),   &getSpacing<itk::Image<float,4>>);
    m.def(("getSpacingVUC2").c_str(), &getSpacing<itk::VectorImage<unsigned char,2>>);
    m.def(("getSpacingVUC3").c_str(), &getSpacing<itk::VectorImage<unsigned char,3>>);
    m.def(("getSpacingVUC4").c_str(), &getSpacing<itk::VectorImage<unsigned char,4>>);
    m.def(("getSpacingVUI2").c_str(), &getSpacing<itk::VectorImage<unsigned int,2>>);
    m.def(("getSpacingVUI3").c_str(), &getSpacing<itk::VectorImage<unsigned int,3>>);
    m.def(("getSpacingVUI4").c_str(), &getSpacing<itk::VectorImage<unsigned int,4>>);
    m.def(("getSpacingVF2").c_str(),  &getSpacing<itk::VectorImage<float,2>>);
    m.def(("getSpacingVF3").c_str(),  &getSpacing<itk::VectorImage<float,3>>);
    m.def(("getSpacingVF4").c_str(),  &getSpacing<itk::VectorImage<float,4>>);
    m.def(("getSpacingRGBUC3").c_str(), &getSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("setSpacingUC2").c_str(),  &setSpacing<itk::Image<unsigned char,2>>);
    m.def(("setSpacingUC3").c_str(),  &setSpacing<itk::Image<unsigned char,3>>);
    m.def(("setSpacingUC4").c_str(),  &setSpacing<itk::Image<unsigned char,4>>);
    m.def(("setSpacingUI2").c_str(),  &setSpacing<itk::Image<unsigned int,2>>);
    m.def(("setSpacingUI3").c_str(),  &setSpacing<itk::Image<unsigned int,3>>);
    m.def(("setSpacingUI4").c_str(),  &setSpacing<itk::Image<unsigned int,4>>);
    m.def(("setSpacingF2").c_str(),   &setSpacing<itk::Image<float,2>>);
    m.def(("setSpacingF3").c_str(),   &setSpacing<itk::Image<float,3>>);
    m.def(("setSpacingF4").c_str(),   &setSpacing<itk::Image<float,4>>);
    m.def(("setSpacingVUC2").c_str(), &setSpacing<itk::VectorImage<unsigned char,2>>);
    m.def(("setSpacingVUC3").c_str(), &setSpacing<itk::VectorImage<unsigned char,3>>);
    m.def(("setSpacingVUC4").c_str(), &setSpacing<itk::VectorImage<unsigned char,4>>);
    m.def(("setSpacingVUI2").c_str(), &setSpacing<itk::VectorImage<unsigned int,2>>);
    m.def(("setSpacingVUI3").c_str(), &setSpacing<itk::VectorImage<unsigned int,3>>);
    m.def(("setSpacingVUI4").c_str(), &setSpacing<itk::VectorImage<unsigned int,4>>);
    m.def(("setSpacingVF2").c_str(),  &setSpacing<itk::VectorImage<float,2>>);
    m.def(("setSpacingVF3").c_str(),  &setSpacing<itk::VectorImage<float,3>>);
    m.def(("setSpacingVF4").c_str(),  &setSpacing<itk::VectorImage<float,4>>);
    m.def(("setSpacingRGBUC3").c_str(), &setSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("getDirectionUC2").c_str(),  &getDirection<itk::Image<unsigned char,2>>);
    m.def(("getDirectionUC3").c_str(),  &getDirection<itk::Image<unsigned char,3>>);
    m.def(("getDirectionUC4").c_str(),  &getDirection<itk::Image<unsigned char,4>>);
    m.def(("getDirectionUI2").c_str(),  &getDirection<itk::Image<unsigned int,2>>);
    m.def(("getDirectionUI3").c_str(),  &getDirection<itk::Image<unsigned int,3>>);
    m.def(("getDirectionUI4").c_str(),  &getDirection<itk::Image<unsigned int,4>>);
    m.def(("getDirectionF2").c_str(),   &getDirection<itk::Image<float,2>>);
    m.def(("getDirectionF3").c_str(),   &getDirection<itk::Image<float,3>>);
    m.def(("getDirectionF4").c_str(),   &getDirection<itk::Image<float,4>>);
    m.def(("getDirectionVUC2").c_str(), &getDirection<itk::VectorImage<unsigned char,2>>);
    m.def(("getDirectionVUC3").c_str(), &getDirection<itk::VectorImage<unsigned char,3>>);
    m.def(("getDirectionVUC4").c_str(), &getDirection<itk::VectorImage<unsigned char,4>>);
    m.def(("getDirectionVUI2").c_str(), &getDirection<itk::VectorImage<unsigned int,2>>);
    m.def(("getDirectionVUI3").c_str(), &getDirection<itk::VectorImage<unsigned int,3>>);
    m.def(("getDirectionVUI4").c_str(), &getDirection<itk::VectorImage<unsigned int,4>>);
    m.def(("getDirectionVF2").c_str(),  &getDirection<itk::VectorImage<float,2>>);
    m.def(("getDirectionVF3").c_str(),  &getDirection<itk::VectorImage<float,3>>);
    m.def(("getDirectionVF4").c_str(),  &getDirection<itk::VectorImage<float,4>>);
    m.def(("getDirectionRGBUC3").c_str(), &getDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("setDirectionUC2").c_str(),  &setDirection<itk::Image<unsigned char,2>>);
    m.def(("setDirectionUC3").c_str(),  &setDirection<itk::Image<unsigned char,3>>);
    m.def(("setDirectionUC4").c_str(),  &setDirection<itk::Image<unsigned char,4>>);
    m.def(("setDirectionUI2").c_str(),  &setDirection<itk::Image<unsigned int,2>>);
    m.def(("setDirectionUI3").c_str(),  &setDirection<itk::Image<unsigned int,3>>);
    m.def(("setDirectionUI4").c_str(),  &setDirection<itk::Image<unsigned int,4>>);
    m.def(("setDirectionF2").c_str(),   &setDirection<itk::Image<float,2>>);
    m.def(("setDirectionF3").c_str(),   &setDirection<itk::Image<float,3>>);
    m.def(("setDirectionF4").c_str(),   &setDirection<itk::Image<float,4>>);
    m.def(("setDirectionVUC2").c_str(), &setDirection<itk::VectorImage<unsigned char,2>>);
    m.def(("setDirectionVUC3").c_str(), &setDirection<itk::VectorImage<unsigned char,3>>);
    m.def(("setDirectionVUC4").c_str(), &setDirection<itk::VectorImage<unsigned char,4>>);
    m.def(("setDirectionVUI2").c_str(), &setDirection<itk::VectorImage<unsigned int,2>>);
    m.def(("setDirectionVUI3").c_str(), &setDirection<itk::VectorImage<unsigned int,3>>);
    m.def(("setDirectionVUI4").c_str(), &setDirection<itk::VectorImage<unsigned int,4>>);
    m.def(("setDirectionVF2").c_str(),  &setDirection<itk::VectorImage<float,2>>);
    m.def(("setDirectionVF3").c_str(),  &setDirection<itk::VectorImage<float,3>>);
    m.def(("setDirectionVF4").c_str(),  &setDirection<itk::VectorImage<float,4>>);
    m.def(("setDirectionRGBUC3").c_str(), &setDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("toFileUC2").c_str(),  &getDirection<itk::Image<unsigned char,2>>);
    m.def(("toFileUC3").c_str(),  &getDirection<itk::Image<unsigned char,3>>);
    m.def(("toFileUC4").c_str(),  &getDirection<itk::Image<unsigned char,4>>);
    m.def(("toFileUI2").c_str(),  &getDirection<itk::Image<unsigned int,2>>);
    m.def(("toFileUI3").c_str(),  &getDirection<itk::Image<unsigned int,3>>);
    m.def(("toFileUI4").c_str(),  &getDirection<itk::Image<unsigned int,4>>);
    m.def(("toFileF2").c_str(),   &getDirection<itk::Image<float,2>>);
    m.def(("toFileF3").c_str(),   &getDirection<itk::Image<float,3>>);
    m.def(("toFileF4").c_str(),   &getDirection<itk::Image<float,4>>);
    m.def(("toFileVUC2").c_str(), &getDirection<itk::VectorImage<unsigned char,2>>);
    m.def(("toFileVUC3").c_str(), &getDirection<itk::VectorImage<unsigned char,3>>);
    m.def(("toFileVUC4").c_str(), &getDirection<itk::VectorImage<unsigned char,4>>);
    m.def(("toFileVUI2").c_str(), &getDirection<itk::VectorImage<unsigned int,2>>);
    m.def(("toFileVUI3").c_str(), &getDirection<itk::VectorImage<unsigned int,3>>);
    m.def(("toFileVUI4").c_str(), &getDirection<itk::VectorImage<unsigned int,4>>);
    m.def(("toFileVF2").c_str(),  &getDirection<itk::VectorImage<float,2>>);
    m.def(("toFileVF3").c_str(),  &getDirection<itk::VectorImage<float,3>>);
    m.def(("toFileVF4").c_str(),  &getDirection<itk::VectorImage<float,4>>);
    m.def(("toFileRGBUC3").c_str(), &getDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);

    m.def(("toFileUC2").c_str(),  &setDirection<itk::Image<unsigned char,2>>);
    m.def(("toFileUC3").c_str(),  &setDirection<itk::Image<unsigned char,3>>);
    m.def(("toFileUC4").c_str(),  &setDirection<itk::Image<unsigned char,4>>);
    m.def(("toFileUI2").c_str(),  &setDirection<itk::Image<unsigned int,2>>);
    m.def(("toFileUI3").c_str(),  &setDirection<itk::Image<unsigned int,3>>);
    m.def(("toFileUI4").c_str(),  &setDirection<itk::Image<unsigned int,4>>);
    m.def(("toFileF2").c_str(),   &setDirection<itk::Image<float,2>>);
    m.def(("toFileF3").c_str(),   &setDirection<itk::Image<float,3>>);
    m.def(("toFileF4").c_str(),   &setDirection<itk::Image<float,4>>);
    m.def(("toFileVUC2").c_str(), &setDirection<itk::VectorImage<unsigned char,2>>);
    m.def(("toFileVUC3").c_str(), &setDirection<itk::VectorImage<unsigned char,3>>);
    m.def(("toFileVUC4").c_str(), &setDirection<itk::VectorImage<unsigned char,4>>);
    m.def(("toFileVUI2").c_str(), &setDirection<itk::VectorImage<unsigned int,2>>);
    m.def(("toFileVUI3").c_str(), &setDirection<itk::VectorImage<unsigned int,3>>);
    m.def(("toFileVUI4").c_str(), &setDirection<itk::VectorImage<unsigned int,4>>);
    m.def(("toFileVF2").c_str(),  &setDirection<itk::VectorImage<float,2>>);
    m.def(("toFileVF3").c_str(),  &setDirection<itk::VectorImage<float,3>>);
    m.def(("toFileVF4").c_str(),  &setDirection<itk::VectorImage<float,4>>);
    m.def(("toFileRGBUC3").c_str(), &setDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);
}


