#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/TileImages.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int TileImages( StrVector instring )
{
    return ants::TileImages(instring, NULL);
}

void wrap_TileImages(nb::module_ &m) {
  m.def("TileImages", &TileImages);
}