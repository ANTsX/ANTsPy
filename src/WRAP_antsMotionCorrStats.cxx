
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsMotionCorrStats.h"

namespace py = pybind11;

int antsMotionCorrStats( std::vector<std::string> instring )
{
    return ants::antsMotionCorrStats(instring, NULL);
}

PYBIND11_MODULE(antsMotionCorrStats, m)
{
  m.def("antsMotionCorrStats", &antsMotionCorrStats);
}
