
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/N3BiasFieldCorrection.h"

namespace py = pybind11;

int N3BiasFieldCorrection( std::vector<std::string> instring )
{
    return ants::N3BiasFieldCorrection(instring, NULL);
}

PYBIND11_MODULE(N3BiasFieldCorrection, m)
{
  m.def("N3BiasFieldCorrection", &N3BiasFieldCorrection);
}