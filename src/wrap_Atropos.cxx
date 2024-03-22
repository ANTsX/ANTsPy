#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

#include "antscore/Atropos.h"

int Atropos( std::vector<std::string> instring )
{
    return ants::Atropos(instring, NULL);
}

void wrap_Atropos(nb::module_ &m) {
  m.def("Atropos", &Atropos);
}

