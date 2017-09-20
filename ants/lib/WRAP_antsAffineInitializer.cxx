
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsAffineInitializer.h"

namespace py = pybind11;

int antsAffineInitializer( std::vector<std::string> instring )
{
    return ants::antsAffineInitializer(instring, NULL);
}

PYBIND11_MODULE(antsAffineInitializer, m)
{
  m.def("antsAffineInitializer", &antsAffineInitializer);
}