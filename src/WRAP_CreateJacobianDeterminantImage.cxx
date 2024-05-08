#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "antscore/CreateJacobianDeterminantImage.h"

namespace nb = nanobind;
using namespace nb::literals;

using StrVector = std::vector<std::string>;

int CreateJacobianDeterminantImage( StrVector instring )
{
    return ants::CreateJacobianDeterminantImage(instring, NULL);
}

void wrap_CreateJacobianDeterminantImage(nb::module_ &m) {
  m.def("CreateJacobianDeterminantImage", &CreateJacobianDeterminantImage);
}