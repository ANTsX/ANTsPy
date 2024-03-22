
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/CreateJacobianDeterminantImage.h"

namespace py = pybind11;

int CreateJacobianDeterminantImage( std::vector<std::string> instring )
{
    return ants::CreateJacobianDeterminantImage(instring, NULL);
}

PYBIND11_MODULE(CreateJacobianDeterminantImage, m)
{
  m.def("CreateJacobianDeterminantImage", &CreateJacobianDeterminantImage);
}