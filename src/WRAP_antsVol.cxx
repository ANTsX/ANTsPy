#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsVol.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsVol( std::vector<std::string> instring )
{
    return ants::antsVol(instring, NULL);
}

void wrap_antsVol(nb::module_ &m) {
  m.def("antsVol", &antsVol);
}