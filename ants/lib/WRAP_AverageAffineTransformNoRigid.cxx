#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/AverageAffineTransformNoRigid.h"

namespace py = pybind11;

int AverageAffineTransformNoRigid( std::vector<std::string> instring )
{
    return ants::AverageAffineTransformNoRigid(instring, NULL);
}

PYBIND11_MODULE(AverageAffineTransformNoRigid, m)
{
  m.def("AverageAffineTransformNoRigid", &AverageAffineTransformNoRigid);
}
