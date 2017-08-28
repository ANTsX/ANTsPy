
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
py::array ANTsImage<ImageType>::numpy()
{
    py::array myArray;
    unsigned int dim = this->dimension;
    std::string pixeltype = this->pixeltype;

    typename ImageType::Pointer itkImage;
    itkImage = as<ImageType>( *this );
    myArray = numpyHelper<ImageType>( itkImage );

    return myArray;
}

std::string ptrstr(py::capsule c)
{
    std::stringstream ss;
    ss << (void const *)c;
    std::string s = ss.str();
    return s;
}



template <typename ImageType, unsigned int ndim>
void wrapANTsImage(py::module & m, std::string const & suffix) {
    py::class_<ANTsImage<ImageType>, std::shared_ptr<ANTsImage<ImageType>>>(m, ("ANTsImage" + suffix).c_str())
        //.def(py::init<>())

        // read only properties
        .def_readonly("pixeltype", &ANTsImage<ImageType>::pixeltype)
        .def_readonly("dtype", &ANTsImage<ImageType>::dtype)
        .def_readonly("dimension", &ANTsImage<ImageType>::dimension)
        .def_readonly("components", &ANTsImage<ImageType>::components)
        .def_readonly("pointer", &ANTsImage<ImageType>::pointer)

        // read-write properties (origin, spacing, direction)
        .def("get_shape", &ANTsImage<ImageType>::getShape)
        .def("get_origin", &ANTsImage<ImageType>::getOrigin)
        .def("set_origin", &ANTsImage<ImageType>::setOrigin)
        .def("get_spacing", &ANTsImage<ImageType>::getSpacing)
        .def("set_spacing", &ANTsImage<ImageType>::setSpacing)
        .def("get_direction", &ANTsImage<ImageType>::getDirection)
        .def("set_direction", &ANTsImage<ImageType>::setDirection)

        // other functions
        .def("toFile", &ANTsImage<ImageType>::toFile)
        .def("numpy", &ANTsImage<ImageType>::numpy);
}



PYBIND11_MODULE(antsImage, m) {
    m.def("ptrstr", &ptrstr);

    wrapANTsImage<itk::Image<unsigned char, 2>,2>(m, "UC2");
    wrapANTsImage<itk::Image<unsigned char, 3>,3>(m, "UC3");
    wrapANTsImage<itk::Image<unsigned char, 4>,4>(m, "UC4");
    wrapANTsImage<itk::Image<unsigned int, 2>,2>(m, "UI2");
    wrapANTsImage<itk::Image<unsigned int, 3>,3>(m, "UI3");
    wrapANTsImage<itk::Image<unsigned int, 4>,4>(m, "UI4");
    wrapANTsImage<itk::Image<float, 2>,2>(m, "F2");
    wrapANTsImage<itk::Image<float, 3>,3>(m, "F3");
    wrapANTsImage<itk::Image<float, 4>,4>(m, "F4");
    wrapANTsImage<itk::Image<double, 2>,2>(m, "D2");
    wrapANTsImage<itk::Image<double, 3>,3>(m, "D3");
    wrapANTsImage<itk::Image<double, 4>,4>(m, "D4");

    wrapANTsImage<itk::VectorImage<unsigned char, 2>,2>(m, "VUC2");
    wrapANTsImage<itk::VectorImage<unsigned char, 3>,3>(m, "VUC3");
    wrapANTsImage<itk::VectorImage<unsigned char, 4>,4>(m, "VUC4");
    wrapANTsImage<itk::VectorImage<unsigned int, 2>,2>(m, "VUI2");
    wrapANTsImage<itk::VectorImage<unsigned int, 3>,3>(m, "VUI3");
    wrapANTsImage<itk::VectorImage<unsigned int, 4>,4>(m, "VUI4");
    wrapANTsImage<itk::VectorImage<float, 2>,2>(m, "VF2");
    wrapANTsImage<itk::VectorImage<float, 3>,3>(m, "VF3");
    wrapANTsImage<itk::VectorImage<float, 4>,4>(m, "VF4");
    wrapANTsImage<itk::VectorImage<double, 2>,2>(m, "VD2");
    wrapANTsImage<itk::VectorImage<double, 3>,3>(m, "VD3");
    wrapANTsImage<itk::VectorImage<double, 4>,4>(m, "VD4");

    wrapANTsImage<itk::Image<itk::RGBPixel<unsigned char>, 3>,3>(m, "RGBUC3");

}


