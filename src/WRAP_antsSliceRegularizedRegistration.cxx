#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsSliceRegularizedRegistration.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsSliceRegularizedRegistration( std::vector<std::string> instring )
{
    return ants::antsSliceRegularizedRegistration(instring, NULL);
}

void wrap_antsSliceRegularizedRegistration(nb::module_ &m) {
  m.def("antsSliceRegularizedRegistration", &antsSliceRegularizedRegistration);
}