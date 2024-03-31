#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/NonLocalSuperResolution.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int NonLocalSuperResolution( std::vector<std::string> instring )
{
    return ants::NonLocalSuperResolution(instring, NULL);
}

void wrap_NonLocalSuperResolution(nb::module_ &m) {
  m.def("NonLocalSuperResolution", &NonLocalSuperResolution);
}