
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/TimeSCCAN.h"

namespace py = pybind11;

int TimeSCCAN( std::vector<std::string> instring )
{
    return ants::TimeSCCAN(instring, NULL);
}

PYBIND11_MODULE(TimeSCCAN, m)
{
  m.def("TimeSCCAN", &TimeSCCAN);
}