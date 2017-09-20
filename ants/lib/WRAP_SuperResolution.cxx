
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/SuperResolution.h"

namespace py = pybind11;

int SuperResolution( std::vector<std::string> instring )
{
    return ants::SuperResolution(instring, NULL);
}

PYBIND11_MODULE(SuperResolution, m)
{
  m.def("SuperResolution", &SuperResolution);
}