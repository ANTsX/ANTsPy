#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsJointFusion.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsJointFusion( std::vector<std::string> instring )
{
    return ants::antsJointFusion(instring, NULL);
}

void wrap_antsJointFusion(nb::module_ &m) {
  m.def("antsJointFusion", &antsJointFusion);
}