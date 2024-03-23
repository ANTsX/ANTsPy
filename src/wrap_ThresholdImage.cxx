#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

#include "antscore/ThresholdImage.h"

int ThresholdImage( std::vector<std::string> instring )
{
    return ants::ThresholdImage(instring, NULL);
}

void wrap_ThresholdImage(nb::module_ &m) {
  m.def("ThresholdImage", &ThresholdImage);
}
