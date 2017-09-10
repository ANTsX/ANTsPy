

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
void destroy_image(ANTsImage<ImageType> antsImage )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType image = ImageType::New();

    image = as<ImageType>( antsImage );

    image = NULL;
}

PYBIND11_MODULE(antsImageDestructor, m)
{
    m.def("destroy_imageF2", &destroy_image<itk::Image<float, 2>>);
    m.def("destroy_imageF3", &destroy_image<itk::Image<float, 3>>);
}