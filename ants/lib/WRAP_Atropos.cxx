
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/Atropos.h"

namespace py = pybind11;

int Atropos( std::vector<std::string> instring )
{
    return ants::Atropos(instring, NULL);
}

PYBIND11_MODULE(Atropos, m)
{
  m.def("Atropos", &Atropos);
}
