#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/CreateTiledMosaic.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int CreateTiledMosaic( std::vector<std::string> instring )
{
    return ants::CreateTiledMosaic(instring, NULL);
}

void wrap_CreateTiledMosaic(nb::module_ &m) {
  m.def("CreateTiledMosaic", &CreateTiledMosaic);
}