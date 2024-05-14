#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/AverageAffineTransform.h"
#include "antscore/AverageAffineTransformNoRigid.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int AverageAffineTransform( StrVector instring )
{
    return ants::AverageAffineTransform(instring, NULL);
}

void wrap_AverageAffineTransform(nb::module_ &m) {
  m.def("AverageAffineTransform", &AverageAffineTransform);
}