#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/DenoiseImage.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int DenoiseImage( StrVector instring )
{
    return ants::DenoiseImage(instring, NULL);
}

void wrap_DenoiseImage(nb::module_ &m) {
  m.def("DenoiseImage", &DenoiseImage);
}