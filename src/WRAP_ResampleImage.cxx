#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/ResampleImage.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int ResampleImage( StrVector instring )
{
    return ants::ResampleImage(instring, NULL);
}

void wrap_ResampleImage(nb::module_ &m) {
  m.def("ResampleImage", &ResampleImage);
}