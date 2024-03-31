#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/LesionFilling.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int LesionFilling( std::vector<std::string> instring )
{
    return ants::LesionFilling(instring, NULL);
}

void wrap_LesionFilling(nb::module_ &m) {
  m.def("LesionFilling", &LesionFilling);
}