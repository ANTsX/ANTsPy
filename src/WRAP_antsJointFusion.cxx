
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsJointFusion.h"

namespace py = pybind11;

int antsJointFusion( std::vector<std::string> instring )
{
    return ants::antsJointFusion(instring, NULL);
}

PYBIND11_MODULE(antsJointFusion, m)
{
  m.def("antsJointFusion", &antsJointFusion);
}