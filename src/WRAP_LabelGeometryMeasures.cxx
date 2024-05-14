#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/LabelGeometryMeasures.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int LabelGeometryMeasures( StrVector instring )
{
    return ants::LabelGeometryMeasures(instring, NULL);
}

void wrap_LabelGeometryMeasures(nb::module_ &m) {
  m.def("LabelGeometryMeasures", &LabelGeometryMeasures);
}