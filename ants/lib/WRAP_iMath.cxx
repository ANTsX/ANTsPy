
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/iMath.h"

namespace py = pybind11;

int iMath( std::vector<std::string> instring )
{
    return ants::iMath(instring, NULL);
}

PYBIND11_MODULE(iMath, m)
{
  m.def("iMath", &iMath);
}
