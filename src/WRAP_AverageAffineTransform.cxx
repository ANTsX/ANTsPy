#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/AverageAffineTransform.h"
#include "antscore/AverageAffineTransformNoRigid.h"

namespace py = pybind11;

int AverageAffineTransform( std::vector<std::string> instring )
{
    return ants::AverageAffineTransform(instring, NULL);
}

PYBIND11_MODULE(AverageAffineTransform, m)
{
  m.def("AverageAffineTransform", &AverageAffineTransform);
}
