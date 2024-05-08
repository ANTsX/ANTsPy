#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/N3BiasFieldCorrection.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int N3BiasFieldCorrection( StrVector instring )
{
    return ants::N3BiasFieldCorrection(instring, NULL);
}

void wrap_N3BiasFieldCorrection(nb::module_ &m) {
  m.def("N3BiasFieldCorrection", &N3BiasFieldCorrection);
}