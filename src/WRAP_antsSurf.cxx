#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsSurf.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsSurf( std::vector<std::string> instring )
{
    return ants::antsSurf(instring, NULL);
}

void wrap_antsSurf(nb::module_ &m) {
  m.def("antsSurf", &antsSurf);
}