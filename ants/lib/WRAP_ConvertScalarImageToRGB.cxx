
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/ConvertScalarImageToRGB.h"

namespace py = pybind11;

int ConvertScalarImageToRGB( std::vector<std::string> instring )
{
    return ants::ConvertScalarImageToRGB(instring, NULL);
}

PYBIND11_MODULE(ConvertScalarImageToRGB, m)
{
  m.def("ConvertScalarImageToRGB", &ConvertScalarImageToRGB);
}