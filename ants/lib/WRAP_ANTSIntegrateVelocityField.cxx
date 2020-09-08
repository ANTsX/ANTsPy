
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/ANTSIntegrateVelocityField.h"

namespace py = pybind11;

int ANTSIntegrateVelocityField( std::vector<std::string> instring )
{
    return ants::ANTSIntegrateVelocityField(instring, NULL);
}

PYBIND11_MODULE(ANTSIntegrateVelocityField, m)
{
  m.def("ANTSIntegrateVelocityField", &ANTSIntegrateVelocityField);
}
