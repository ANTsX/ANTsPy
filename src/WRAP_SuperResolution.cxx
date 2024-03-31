#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/SuperResolution.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int SuperResolution( std::vector<std::string> instring )
{
    return ants::SuperResolution(instring, NULL);
}

void wrap_SuperResolution(nb::module_ &m) {
  m.def("SuperResolution", &SuperResolution);
}