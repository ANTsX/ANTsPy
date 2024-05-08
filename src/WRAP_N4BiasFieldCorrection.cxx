#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/N4BiasFieldCorrection.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int N4BiasFieldCorrection( StrVector instring )
{
    return ants::N4BiasFieldCorrection(instring, NULL);
}

void wrap_N4BiasFieldCorrection(nb::module_ &m) {
  m.def("N4BiasFieldCorrection", &N4BiasFieldCorrection);
}