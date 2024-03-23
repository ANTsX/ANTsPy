#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsRegistration.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsRegistration( StrVector instring )
{
    return ants::antsRegistration(instring, NULL);
}

void wrap_antsRegistration(nb::module_ &m) {
  m.def("antsRegistration", &antsRegistration);
}

