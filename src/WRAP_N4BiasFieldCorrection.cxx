
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/N4BiasFieldCorrection.h"

namespace py = pybind11;

int N4BiasFieldCorrection( std::vector<std::string> instring )
{
    return ants::N4BiasFieldCorrection(instring, NULL);
}

PYBIND11_MODULE(N4BiasFieldCorrection, m)
{
  m.def("N4BiasFieldCorrection", &N4BiasFieldCorrection);
}