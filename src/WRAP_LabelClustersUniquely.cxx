#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/LabelClustersUniquely.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int LabelClustersUniquely( StrVector instring )
{
    return ants::LabelClustersUniquely(instring, NULL);
}

void wrap_LabelClustersUniquely(nb::module_ &m) {
  m.def("LabelClustersUniquely", &LabelClustersUniquely);
}