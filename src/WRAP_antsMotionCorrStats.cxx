#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsMotionCorrStats.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsMotionCorrStats( std::vector<std::string> instring )
{
    return ants::antsMotionCorrStats(instring, NULL);
}

void wrap_antsMotionCorrStats(nb::module_ &m) {
  m.def("antsMotionCorrStats", &antsMotionCorrStats);
}