
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsApplyTransforms.h"

namespace py = pybind11;

int antsApplyTransforms( std::vector<std::string> instring )
{
    return ants::antsApplyTransforms(instring, NULL);
}

PYBIND11_MODULE(antsApplyTransforms, m)
{
  m.def("antsApplyTransforms", &antsApplyTransforms);
}