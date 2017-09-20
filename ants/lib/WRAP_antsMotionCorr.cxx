
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsMotionCorr.h"

namespace py = pybind11;

int antsMotionCorr( std::vector<std::string> instring )
{
    return ants::antsMotionCorr(instring, NULL);
}

PYBIND11_MODULE(antsMotionCorr, m)
{
  m.def("antsMotionCorr", &antsMotionCorr);
}
