#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/Atropos.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int Atropos( StrVector instring )
{
    return ants::Atropos(instring, NULL);
}

void wrap_Atropos(nb::module_ &m) {
  m.def("Atropos", &Atropos);
}

