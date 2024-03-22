#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsAffineInitializer.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsAffineInitializer( StrVector instring )
{
    return ants::antsAffineInitializer(instring, NULL);
}

void wrap_antsAffineInitializer(nb::module_ &m) {
  m.def("antsAffineInitializer", &antsAffineInitializer);
}

