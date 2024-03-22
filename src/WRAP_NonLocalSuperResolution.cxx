
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/NonLocalSuperResolution.h"

namespace py = pybind11;

int NonLocalSuperResolution( std::vector<std::string> instring )
{
    return ants::NonLocalSuperResolution(instring, NULL);
}

PYBIND11_MODULE(NonLocalSuperResolution, m)
{
  m.def("NonLocalSuperResolution", &NonLocalSuperResolution);
}