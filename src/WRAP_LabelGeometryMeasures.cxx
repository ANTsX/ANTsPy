
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/LabelGeometryMeasures.h"

namespace py = pybind11;

int LabelGeometryMeasures( std::vector<std::string> instring )
{
    return ants::LabelGeometryMeasures(instring, NULL);
}

PYBIND11_MODULE(LabelGeometryMeasures, m)
{
  m.def("LabelGeometryMeasures", &LabelGeometryMeasures);
}