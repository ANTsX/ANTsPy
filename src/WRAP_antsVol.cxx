
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsVol.h"

namespace py = pybind11;

int antsVol( std::vector<std::string> instring )
{
    return ants::antsVol(instring, NULL);
}

PYBIND11_MODULE(antsVol, m)
{
  m.def("antsVol", &antsVol);
}