
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
#include "antsImage.h"


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

/** original version
std::string ptrstr(py::capsule c)
{
    std::stringstream ss;
    ss << (void const *)c;
    std::string s = ss.str();
    return s;
}
*/

// version contributed by @SGotla
std::string ptrstr(void * c)
{
	char buffer[50];
	int n = sprintf(buffer, "0x%p", (void const *)c);
	std::string s(buffer, n);
	return s;
}




PYBIND11_MODULE(antsImage, m) {
    m.def("ptrstr", &ptrstr);

    m.def("getShapeUC2",  &getShape<itk::Image<unsigned char,2>>);
    m.def("getShapeUC3",  &getShape<itk::Image<unsigned char,3>>);
    m.def("getShapeUC4",  &getShape<itk::Image<unsigned char,4>>);
    m.def("getShapeUI2",  &getShape<itk::Image<unsigned int,2>>);
    m.def("getShapeUI3",  &getShape<itk::Image<unsigned int,3>>);
    m.def("getShapeUI4",  &getShape<itk::Image<unsigned int,4>>);
    m.def("getShapeF2",   &getShape<itk::Image<float,2>>);
    m.def("getShapeF3",   &getShape<itk::Image<float,3>>);
    m.def("getShapeF4",   &getShape<itk::Image<float,4>>);
    m.def("getShapeD2",   &getShape<itk::Image<double,2>>);
    m.def("getShapeD3",   &getShape<itk::Image<double,3>>);
    m.def("getShapeD4",   &getShape<itk::Image<double,4>>);
    m.def("getShapeVUC2", &getShape<itk::VectorImage<unsigned char,2>>);
    m.def("getShapeVUC3", &getShape<itk::VectorImage<unsigned char,3>>);
    m.def("getShapeVUC4", &getShape<itk::VectorImage<unsigned char,4>>);
    m.def("getShapeVUI2", &getShape<itk::VectorImage<unsigned int,2>>);
    m.def("getShapeVUI3", &getShape<itk::VectorImage<unsigned int,3>>);
    m.def("getShapeVUI4", &getShape<itk::VectorImage<unsigned int,4>>);
    m.def("getShapeVF2",  &getShape<itk::VectorImage<float,2>>);
    m.def("getShapeVF3",  &getShape<itk::VectorImage<float,3>>);
    m.def("getShapeVF4",  &getShape<itk::VectorImage<float,4>>);
    m.def("getShapeVD2",  &getShape<itk::VectorImage<double,2>>);
    m.def("getShapeVD3",  &getShape<itk::VectorImage<double,3>>);
    m.def("getShapeVD4",  &getShape<itk::VectorImage<double,4>>);
    m.def("getShapeRGBUC2", &getShape<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getShapeRGBUC3", &getShape<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getShapeRGBF2", &getShape<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getShapeRGBF3", &getShape<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getOriginUC2",  &getOrigin<itk::Image<unsigned char,2>>);
    m.def("getOriginUC3",  &getOrigin<itk::Image<unsigned char,3>>);
    m.def("getOriginUC4",  &getOrigin<itk::Image<unsigned char,4>>);
    m.def("getOriginUI2",  &getOrigin<itk::Image<unsigned int,2>>);
    m.def("getOriginUI3",  &getOrigin<itk::Image<unsigned int,3>>);
    m.def("getOriginUI4",  &getOrigin<itk::Image<unsigned int,4>>);
    m.def("getOriginF2",   &getOrigin<itk::Image<float,2>>);
    m.def("getOriginF3",   &getOrigin<itk::Image<float,3>>);
    m.def("getOriginF4",   &getOrigin<itk::Image<float,4>>);
    m.def("getOriginD2",   &getOrigin<itk::Image<double,2>>);
    m.def("getOriginD3",   &getOrigin<itk::Image<double,3>>);
    m.def("getOriginD4",   &getOrigin<itk::Image<double,4>>);
    m.def("getOriginVUC2", &getOrigin<itk::VectorImage<unsigned char,2>>);
    m.def("getOriginVUC3", &getOrigin<itk::VectorImage<unsigned char,3>>);
    m.def("getOriginVUC4", &getOrigin<itk::VectorImage<unsigned char,4>>);
    m.def("getOriginVUI2", &getOrigin<itk::VectorImage<unsigned int,2>>);
    m.def("getOriginVUI3", &getOrigin<itk::VectorImage<unsigned int,3>>);
    m.def("getOriginVUI4", &getOrigin<itk::VectorImage<unsigned int,4>>);
    m.def("getOriginVF2",  &getOrigin<itk::VectorImage<float,2>>);
    m.def("getOriginVF3",  &getOrigin<itk::VectorImage<float,3>>);
    m.def("getOriginVF4",  &getOrigin<itk::VectorImage<float,4>>);
    m.def("getOriginVD2",  &getOrigin<itk::VectorImage<double,2>>);
    m.def("getOriginVD3",  &getOrigin<itk::VectorImage<double,3>>);
    m.def("getOriginVD4",  &getOrigin<itk::VectorImage<double,4>>);
    m.def("getOriginRGBUC2", &getOrigin<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getOriginRGBUC3", &getOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getOriginRGBF2", &getOrigin<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getOriginRGBF3", &getOrigin<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setOriginUC2",  &setOrigin<itk::Image<unsigned char,2>>);
    m.def("setOriginUC3",  &setOrigin<itk::Image<unsigned char,3>>);
    m.def("setOriginUC4",  &setOrigin<itk::Image<unsigned char,4>>);
    m.def("setOriginUI2",  &setOrigin<itk::Image<unsigned int,2>>);
    m.def("setOriginUI3",  &setOrigin<itk::Image<unsigned int,3>>);
    m.def("setOriginUI4",  &setOrigin<itk::Image<unsigned int,4>>);
    m.def("setOriginF2",   &setOrigin<itk::Image<float,2>>);
    m.def("setOriginF3",   &setOrigin<itk::Image<float,3>>);
    m.def("setOriginF4",   &setOrigin<itk::Image<float,4>>);
    m.def("setOriginD2",   &setOrigin<itk::Image<double,2>>);
    m.def("setOriginD3",   &setOrigin<itk::Image<double,3>>);
    m.def("setOriginD4",   &setOrigin<itk::Image<double,4>>);
    m.def("setOriginVUC2", &setOrigin<itk::VectorImage<unsigned char,2>>);
    m.def("setOriginVUC3", &setOrigin<itk::VectorImage<unsigned char,3>>);
    m.def("setOriginVUC4", &setOrigin<itk::VectorImage<unsigned char,4>>);
    m.def("setOriginVUI2", &setOrigin<itk::VectorImage<unsigned int,2>>);
    m.def("setOriginVUI3", &setOrigin<itk::VectorImage<unsigned int,3>>);
    m.def("setOriginVUI4", &setOrigin<itk::VectorImage<unsigned int,4>>);
    m.def("setOriginVF2",  &setOrigin<itk::VectorImage<float,2>>);
    m.def("setOriginVF3",  &setOrigin<itk::VectorImage<float,3>>);
    m.def("setOriginVF4",  &setOrigin<itk::VectorImage<float,4>>);
    m.def("setOriginVD2",  &setOrigin<itk::VectorImage<double,2>>);
    m.def("setOriginVD3",  &setOrigin<itk::VectorImage<double,3>>);
    m.def("setOriginVD4",  &setOrigin<itk::VectorImage<double,4>>);
    m.def("setOriginRGBUC2", &setOrigin<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setOriginRGBUC3", &setOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setOriginRGBF2", &setOrigin<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setOriginRGBF3", &setOrigin<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getSpacingUC2",  &getSpacing<itk::Image<unsigned char,2>>);
    m.def("getSpacingUC3",  &getSpacing<itk::Image<unsigned char,3>>);
    m.def("getSpacingUC4",  &getSpacing<itk::Image<unsigned char,4>>);
    m.def("getSpacingUI2",  &getSpacing<itk::Image<unsigned int,2>>);
    m.def("getSpacingUI3",  &getSpacing<itk::Image<unsigned int,3>>);
    m.def("getSpacingUI4",  &getSpacing<itk::Image<unsigned int,4>>);
    m.def("getSpacingF2",   &getSpacing<itk::Image<float,2>>);
    m.def("getSpacingF3",   &getSpacing<itk::Image<float,3>>);
    m.def("getSpacingF4",   &getSpacing<itk::Image<float,4>>);
    m.def("getSpacingD2",   &getSpacing<itk::Image<double,2>>);
    m.def("getSpacingD3",   &getSpacing<itk::Image<double,3>>);
    m.def("getSpacingD4",   &getSpacing<itk::Image<double,4>>);
    m.def("getSpacingVUC2", &getSpacing<itk::VectorImage<unsigned char,2>>);
    m.def("getSpacingVUC3", &getSpacing<itk::VectorImage<unsigned char,3>>);
    m.def("getSpacingVUC4", &getSpacing<itk::VectorImage<unsigned char,4>>);
    m.def("getSpacingVUI2", &getSpacing<itk::VectorImage<unsigned int,2>>);
    m.def("getSpacingVUI3", &getSpacing<itk::VectorImage<unsigned int,3>>);
    m.def("getSpacingVUI4", &getSpacing<itk::VectorImage<unsigned int,4>>);
    m.def("getSpacingVF2",  &getSpacing<itk::VectorImage<float,2>>);
    m.def("getSpacingVF3",  &getSpacing<itk::VectorImage<float,3>>);
    m.def("getSpacingVF4",  &getSpacing<itk::VectorImage<float,4>>);
    m.def("getSpacingVD2",  &getSpacing<itk::VectorImage<double,2>>);
    m.def("getSpacingVD3",  &getSpacing<itk::VectorImage<double,3>>);
    m.def("getSpacingVD4",  &getSpacing<itk::VectorImage<double,4>>);
    m.def("getSpacingRGBUC2", &getSpacing<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getSpacingRGBUC3", &getSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getSpacingRGBF2", &getSpacing<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getSpacingRGBF3", &getSpacing<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setSpacingUC2",  &setSpacing<itk::Image<unsigned char,2>>);
    m.def("setSpacingUC3",  &setSpacing<itk::Image<unsigned char,3>>);
    m.def("setSpacingUC4",  &setSpacing<itk::Image<unsigned char,4>>);
    m.def("setSpacingUI2",  &setSpacing<itk::Image<unsigned int,2>>);
    m.def("setSpacingUI3",  &setSpacing<itk::Image<unsigned int,3>>);
    m.def("setSpacingUI4",  &setSpacing<itk::Image<unsigned int,4>>);
    m.def("setSpacingF2",   &setSpacing<itk::Image<float,2>>);
    m.def("setSpacingF3",   &setSpacing<itk::Image<float,3>>);
    m.def("setSpacingF4",   &setSpacing<itk::Image<float,4>>);
    m.def("setSpacingD2",   &setSpacing<itk::Image<double,2>>);
    m.def("setSpacingD3",   &setSpacing<itk::Image<double,3>>);
    m.def("setSpacingD4",   &setSpacing<itk::Image<double,4>>);
    m.def("setSpacingVUC2", &setSpacing<itk::VectorImage<unsigned char,2>>);
    m.def("setSpacingVUC3", &setSpacing<itk::VectorImage<unsigned char,3>>);
    m.def("setSpacingVUC4", &setSpacing<itk::VectorImage<unsigned char,4>>);
    m.def("setSpacingVUI2", &setSpacing<itk::VectorImage<unsigned int,2>>);
    m.def("setSpacingVUI3", &setSpacing<itk::VectorImage<unsigned int,3>>);
    m.def("setSpacingVUI4", &setSpacing<itk::VectorImage<unsigned int,4>>);
    m.def("setSpacingVF2",  &setSpacing<itk::VectorImage<float,2>>);
    m.def("setSpacingVF3",  &setSpacing<itk::VectorImage<float,3>>);
    m.def("setSpacingVF4",  &setSpacing<itk::VectorImage<float,4>>);
    m.def("setSpacingVD2",  &setSpacing<itk::VectorImage<double,2>>);
    m.def("setSpacingVD3",  &setSpacing<itk::VectorImage<double,3>>);
    m.def("setSpacingVD4",  &setSpacing<itk::VectorImage<double,4>>);
    m.def("setSpacingRGBUC2", &setSpacing<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setSpacingRGBUC3", &setSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setSpacingRGBF2", &setSpacing<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setSpacingRGBF3", &setSpacing<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getDirectionUC2",  &getDirection<itk::Image<unsigned char,2>>);
    m.def("getDirectionUC3",  &getDirection<itk::Image<unsigned char,3>>);
    m.def("getDirectionUC4",  &getDirection<itk::Image<unsigned char,4>>);
    m.def("getDirectionUI2",  &getDirection<itk::Image<unsigned int,2>>);
    m.def("getDirectionUI3",  &getDirection<itk::Image<unsigned int,3>>);
    m.def("getDirectionUI4",  &getDirection<itk::Image<unsigned int,4>>);
    m.def("getDirectionF2",   &getDirection<itk::Image<float,2>>);
    m.def("getDirectionF3",   &getDirection<itk::Image<float,3>>);
    m.def("getDirectionF4",   &getDirection<itk::Image<float,4>>);
    m.def("getDirectionD2",   &getDirection<itk::Image<double,2>>);
    m.def("getDirectionD3",   &getDirection<itk::Image<double,3>>);
    m.def("getDirectionD4",   &getDirection<itk::Image<double,4>>);
    m.def("getDirectionVUC2", &getDirection<itk::VectorImage<unsigned char,2>>);
    m.def("getDirectionVUC3", &getDirection<itk::VectorImage<unsigned char,3>>);
    m.def("getDirectionVUC4", &getDirection<itk::VectorImage<unsigned char,4>>);
    m.def("getDirectionVUI2", &getDirection<itk::VectorImage<unsigned int,2>>);
    m.def("getDirectionVUI3", &getDirection<itk::VectorImage<unsigned int,3>>);
    m.def("getDirectionVUI4", &getDirection<itk::VectorImage<unsigned int,4>>);
    m.def("getDirectionVF2",  &getDirection<itk::VectorImage<float,2>>);
    m.def("getDirectionVF3",  &getDirection<itk::VectorImage<float,3>>);
    m.def("getDirectionVF4",  &getDirection<itk::VectorImage<float,4>>);
    m.def("getDirectionVD2",  &getDirection<itk::VectorImage<double,2>>);
    m.def("getDirectionVD3",  &getDirection<itk::VectorImage<double,3>>);
    m.def("getDirectionVD4",  &getDirection<itk::VectorImage<double,4>>);
    m.def("getDirectionRGBUC2", &getDirection<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getDirectionRGBUC3", &getDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getDirectionRGBF2", &getDirection<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getDirectionRGBF3", &getDirection<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setDirectionUC2",  &setDirection<itk::Image<unsigned char,2>>);
    m.def("setDirectionUC3",  &setDirection<itk::Image<unsigned char,3>>);
    m.def("setDirectionUC4",  &setDirection<itk::Image<unsigned char,4>>);
    m.def("setDirectionUI2",  &setDirection<itk::Image<unsigned int,2>>);
    m.def("setDirectionUI3",  &setDirection<itk::Image<unsigned int,3>>);
    m.def("setDirectionUI4",  &setDirection<itk::Image<unsigned int,4>>);
    m.def("setDirectionF2",   &setDirection<itk::Image<float,2>>);
    m.def("setDirectionF3",   &setDirection<itk::Image<float,3>>);
    m.def("setDirectionF4",   &setDirection<itk::Image<float,4>>);
    m.def("setDirectionD2",   &setDirection<itk::Image<double,2>>);
    m.def("setDirectionD3",   &setDirection<itk::Image<double,3>>);
    m.def("setDirectionD4",   &setDirection<itk::Image<double,4>>);
    m.def("setDirectionVUC2", &setDirection<itk::VectorImage<unsigned char,2>>);
    m.def("setDirectionVUC3", &setDirection<itk::VectorImage<unsigned char,3>>);
    m.def("setDirectionVUC4", &setDirection<itk::VectorImage<unsigned char,4>>);
    m.def("setDirectionVUI2", &setDirection<itk::VectorImage<unsigned int,2>>);
    m.def("setDirectionVUI3", &setDirection<itk::VectorImage<unsigned int,3>>);
    m.def("setDirectionVUI4", &setDirection<itk::VectorImage<unsigned int,4>>);
    m.def("setDirectionVF2",  &setDirection<itk::VectorImage<float,2>>);
    m.def("setDirectionVF3",  &setDirection<itk::VectorImage<float,3>>);
    m.def("setDirectionVF4",  &setDirection<itk::VectorImage<float,4>>);
    m.def("setDirectionVD2",  &setDirection<itk::VectorImage<double,2>>);
    m.def("setDirectionVD3",  &setDirection<itk::VectorImage<double,3>>);
    m.def("setDirectionVD4",  &setDirection<itk::VectorImage<double,4>>);
    m.def("setDirectionRGBUC2", &setDirection<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setDirectionRGBUC3", &setDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setDirectionRGBF2", &setDirection<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setDirectionRGBF3", &setDirection<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("toFileUC2",  &toFile<itk::Image<unsigned char,2>>);
    m.def("toFileUC3",  &toFile<itk::Image<unsigned char,3>>);
    m.def("toFileUC4",  &toFile<itk::Image<unsigned char,4>>);
    m.def("toFileUI2",  &toFile<itk::Image<unsigned int,2>>);
    m.def("toFileUI3",  &toFile<itk::Image<unsigned int,3>>);
    m.def("toFileUI4",  &toFile<itk::Image<unsigned int,4>>);
    m.def("toFileF2",   &toFile<itk::Image<float,2>>);
    m.def("toFileF3",   &toFile<itk::Image<float,3>>);
    m.def("toFileF4",   &toFile<itk::Image<float,4>>);
    m.def("toFileD2",   &toFile<itk::Image<double,2>>);
    m.def("toFileD3",   &toFile<itk::Image<double,3>>);
    m.def("toFileD4",   &toFile<itk::Image<double,4>>);
    m.def("toFileVUC2", &toFile<itk::VectorImage<unsigned char,2>>);
    m.def("toFileVUC3", &toFile<itk::VectorImage<unsigned char,3>>);
    m.def("toFileVUC4", &toFile<itk::VectorImage<unsigned char,4>>);
    m.def("toFileVUI2", &toFile<itk::VectorImage<unsigned int,2>>);
    m.def("toFileVUI3", &toFile<itk::VectorImage<unsigned int,3>>);
    m.def("toFileVUI4", &toFile<itk::VectorImage<unsigned int,4>>);
    m.def("toFileVF2",  &toFile<itk::VectorImage<float,2>>);
    m.def("toFileVF3",  &toFile<itk::VectorImage<float,3>>);
    m.def("toFileVF4",  &toFile<itk::VectorImage<float,4>>);
    m.def("toFileVD2",  &toFile<itk::VectorImage<double,2>>);
    m.def("toFileVD3",  &toFile<itk::VectorImage<double,3>>);
    m.def("toFileVD4",  &toFile<itk::VectorImage<double,4>>);
    m.def("toFileRGBUC2", &toFile<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("toFileRGBUC3", &toFile<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("toFileRBGF2", &toFile<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("toFileRBGF3", &toFile<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("toNumpyUC2",  &toNumpy<itk::Image<unsigned char,2>>);
    m.def("toNumpyUC3",  &toNumpy<itk::Image<unsigned char,3>>);
    m.def("toNumpyUC4",  &toNumpy<itk::Image<unsigned char,4>>);
    m.def("toNumpyUI2",  &toNumpy<itk::Image<unsigned int,2>>);
    m.def("toNumpyUI3",  &toNumpy<itk::Image<unsigned int,3>>);
    m.def("toNumpyUI4",  &toNumpy<itk::Image<unsigned int,4>>);
    m.def("toNumpyF2",   &toNumpy<itk::Image<float,2>>);
    m.def("toNumpyF3",   &toNumpy<itk::Image<float,3>>);
    m.def("toNumpyF4",   &toNumpy<itk::Image<float,4>>);
    m.def("toNumpyD2",   &toNumpy<itk::Image<double,2>>);
    m.def("toNumpyD3",   &toNumpy<itk::Image<double,3>>);
    m.def("toNumpyD4",   &toNumpy<itk::Image<double,4>>);
    m.def("toNumpyVUC2", &toNumpy<itk::VectorImage<unsigned char,2>>);
    m.def("toNumpyVUC3", &toNumpy<itk::VectorImage<unsigned char,3>>);
    m.def("toNumpyVUC4", &toNumpy<itk::VectorImage<unsigned char,4>>);
    m.def("toNumpyVUI2", &toNumpy<itk::VectorImage<unsigned int,2>>);
    m.def("toNumpyVUI3", &toNumpy<itk::VectorImage<unsigned int,3>>);
    m.def("toNumpyVUI4", &toNumpy<itk::VectorImage<unsigned int,4>>);
    m.def("toNumpyVF2",  &toNumpy<itk::VectorImage<float,2>>);
    m.def("toNumpyVF3",  &toNumpy<itk::VectorImage<float,3>>);
    m.def("toNumpyVF4",  &toNumpy<itk::VectorImage<float,4>>);
    m.def("toNumpyVD2",  &toNumpy<itk::VectorImage<double,2>>);
    m.def("toNumpyVD3",  &toNumpy<itk::VectorImage<double,3>>);
    m.def("toNumpyVD4",  &toNumpy<itk::VectorImage<double,4>>);
    m.def("toNumpyRGBUC2", &toNumpy<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("toNumpyRGBUC3", &toNumpy<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("toNumpyRGBF2", &toNumpy<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("toNumpyRGBF3", &toNumpy<itk::Image<itk::RGBPixel<float>,3>>);

}
