
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsSurf.h"

namespace py = pybind11;

int antsSurf( std::vector<std::string> instring )
{
    return ants::antsSurf(instring, NULL);
}

PYBIND11_MODULE(antsSurf, m)
{
  m.def("antsSurf", &antsSurf);
}