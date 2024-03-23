
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/LesionFilling.h"

namespace py = pybind11;

int LesionFilling( std::vector<std::string> instring )
{
    return ants::LesionFilling(instring, NULL);
}

PYBIND11_MODULE(LesionFilling, m)
{
  m.def("LesionFilling", &LesionFilling);
}