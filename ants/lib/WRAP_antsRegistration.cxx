
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsRegistration.h"

namespace py = pybind11;

int antsRegistration( std::vector<std::string> instring )
{
    return ants::antsRegistration(instring, NULL);
}

PYBIND11_MODULE(antsRegistration, m)
{
  m.def("antsRegistration", &antsRegistration);
}