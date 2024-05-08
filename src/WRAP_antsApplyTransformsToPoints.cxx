#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsApplyTransformsToPoints.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsApplyTransformsToPoints( std::vector<std::string> instring )
{
    return ants::antsApplyTransformsToPoints(instring, NULL);
}

void wrap_antsApplyTransformsToPoints(nb::module_ &m) {
  m.def("antsApplyTransformsToPoints", &antsApplyTransformsToPoints);
}