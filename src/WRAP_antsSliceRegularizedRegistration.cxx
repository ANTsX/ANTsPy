
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsSliceRegularizedRegistration.h"

namespace py = pybind11;

int antsSliceRegularizedRegistration( std::vector<std::string> instring )
{
    return ants::antsSliceRegularizedRegistration(instring, NULL);
}

PYBIND11_MODULE(antsSliceRegularizedRegistration, m)
{
  m.def("antsSliceRegularizedRegistration", &antsSliceRegularizedRegistration);
}