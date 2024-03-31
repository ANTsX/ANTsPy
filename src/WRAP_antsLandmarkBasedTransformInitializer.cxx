#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/antsLandmarkBasedTransformInitializer.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int antsLandmarkBasedTransformInitializer( std::vector<std::string> instring )
{
    return ants::antsLandmarkBasedTransformInitializer(instring, NULL);
}

void wrap_antsLandmarkBasedTransformInitializer(nb::module_ &m) {
  m.def("antsLandmarkBasedTransformInitializer", &antsLandmarkBasedTransformInitializer);
}