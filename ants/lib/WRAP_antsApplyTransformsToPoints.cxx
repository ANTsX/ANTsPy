
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsApplyTransformsToPoints.h"

namespace py = pybind11;

int antsApplyTransformsToPoints( std::vector<std::string> instring )
{
    return ants::antsApplyTransformsToPoints(instring, NULL);
}

PYBIND11_MODULE(antsApplyTransformsToPoints, m)
{
  m.def("antsApplyTransformsToPoints", &antsApplyTransformsToPoints);
}
