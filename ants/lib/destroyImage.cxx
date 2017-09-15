

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antsImage.h"
#include "itkImage.h"


void destroyImage(ANTsImage<itk::Image<float,3>> ants_image)
{
    typedef itk::Image<float,3> ImageType;
    typedef typename ImageType::Pointer ImagePointerType;

    ImagePointerType ptr = as<ImageType>(ants_image);
    ptr = ITK_NULLPTR;
    //delete ptr;
    //delete ptr;

}

PYBIND11_MODULE(destroyImage, m)
{
    m.def("destroyImage", &destroyImage);
}