#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/TimeSCCAN.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int TimeSCCAN( std::vector<std::string> instring )
{
    return ants::TimeSCCAN(instring, NULL);
}

void wrap_TimeSCCAN(nb::module_ &m) {
  m.def("TimeSCCAN", &TimeSCCAN);
}