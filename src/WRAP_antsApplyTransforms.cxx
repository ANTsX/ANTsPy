#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsApplyTransforms.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsApplyTransforms( std::vector<std::string> instring )
{
    return ants::antsApplyTransforms(instring, NULL);
}

void wrap_antsApplyTransforms(nb::module_ &m) {
  m.def("antsApplyTransforms", &antsApplyTransforms);
}