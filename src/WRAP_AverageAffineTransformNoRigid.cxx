#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/AverageAffineTransformNoRigid.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int AverageAffineTransformNoRigid( StrVector instring )
{
    return ants::AverageAffineTransformNoRigid(instring, NULL);
}

void wrap_AverageAffineTransformNoRigid(nb::module_ &m) {
  m.def("AverageAffineTransformNoRigid", &AverageAffineTransformNoRigid);
}