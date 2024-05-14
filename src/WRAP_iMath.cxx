#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/iMath.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int iMath( StrVector instring )
{
    return ants::iMath(instring, NULL);
}

void wrap_iMath(nb::module_ &m) {
  m.def("iMath", &iMath);
}