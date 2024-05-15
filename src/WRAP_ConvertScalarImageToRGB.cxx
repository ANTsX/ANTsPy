#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/ConvertScalarImageToRGB.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int ConvertScalarImageToRGB( StrVector instring )
{
    return ants::ConvertScalarImageToRGB(instring, NULL);
}

void wrap_ConvertScalarImageToRGB(nb::module_ &m) {
  m.def("ConvertScalarImageToRGB", &ConvertScalarImageToRGB);
}