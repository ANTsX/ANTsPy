#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/KellyKapowski.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int KellyKapowski( StrVector instring )
{
    return ants::KellyKapowski(instring, NULL);
}

void wrap_KellyKapowski(nb::module_ &m) {
  m.def("KellyKapowski", &KellyKapowski);
}