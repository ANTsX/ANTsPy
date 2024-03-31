#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsMotionCorr.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsMotionCorr( std::vector<std::string> instring )
{
    return ants::antsMotionCorr(instring, NULL);
}

void wrap_antsMotionCorr(nb::module_ &m) {
  m.def("antsMotionCorr", &antsMotionCorr);
}